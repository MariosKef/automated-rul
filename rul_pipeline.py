import multiprocessing
import os
import pickle as pkl
import sys
import time

import numpy as np
import pandas as pd
import tpot.config
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tpot import TPOTRegressor
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

del tpot.config.regressor_config_dict['xgboost.XGBRegressor']


# Preprocessing the dataset
def preprocessing(cfg, t, n_jobs):
    """
    This function takes loads raw sensor data
    and the ground truth RUL values of the test
    data standardizes them, performs the expanding
    window transformation over the training set,
    creates the RUL labels in a piece-wise linear
    manner and extracts statistics from the expanded
    windows of the training set and from the entirety
    of the test set.

    :param cfg: configuration file (type=[]) with the window size (index 0)
    and the initial constant RUL value (index 1)
    :param t: dataset number (1-4)
    :param n_jobs: number of parallel jobs
    :return:
        X: The statistical features of all expanded windows of the training set
        Y: The RUL of every sample in X
        extracted_features_test: The statistical features of all expanded windows of the test set
        RUL: The ground truth RUL values of the test set

    """
    train = pd.read_csv('./Data/Data_set' + str(t) + '/train_FD00' + str(t) + '.csv', parse_dates=False, delimiter=" ",
                        decimal=".",
                        header=None)
    train.drop(train.columns[[-1, -2]], axis=1, inplace=True)
    cols = ['unit', 'cycles', 'op_setting1', 'op_setting2', 'op_setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7',
            's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    train.columns = cols

    scaler = StandardScaler()
    train_sensors = scaler.fit_transform(train.iloc[:, 2:])
    train_scaled = pd.DataFrame(columns=cols)
    train_scaled['unit'] = train.unit.values
    train_scaled['cycles'] = train.cycles.values
    train_scaled.iloc[:, 2:] = train_sensors

    # Sensors/features to keep
    keep = [0, 1, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]  # 0,1 are units and cycles, respt.

    if t in [1, 3]:
        train = train_scaled.iloc[:, keep]
    else:
        train = train_scaled

    test = pd.read_csv('./Data/Data_set' + str(t) + '/test_FD00' + str(t) + '.csv', parse_dates=False, delimiter=" ",
                       decimal=".",
                       header=None)
    test.drop(test.columns[[-1, -2]], axis=1, inplace=True)

    test.columns = cols

    test_sensors = scaler.transform(test.iloc[:, 2:])
    test_scaled = pd.DataFrame(columns=cols)
    test_scaled['unit'] = test.unit.values
    test_scaled['cycles'] = test.cycles.values
    test_scaled.iloc[:, 2:] = test_sensors

    # Here we check if t belongs to FD001 or FD003
    if t in [1, 3]:
        test = test_scaled.iloc[:, keep]
    else:
        test = test_scaled

    print(train.columns)
    print(test.columns)

    RUL = pd.read_csv('./Data/Data_set' + str(t) + '/RUL_FD00' + str(t) + '.csv', parse_dates=False, delimiter=" ",
                      decimal=".", header=None)
    RUL.drop(RUL.columns[[-1]], axis=1, inplace=True)
    RUL.index = range(1, len(RUL) + 1)

    cols = train.columns
    df = train

    print(df.columns)

    print(f'Train shape: {train.shape}')
    print(f'Test shape: {test.shape}')
    print(f'RUL shape: {RUL.shape}')

    subsequences = {}
    for unit in df.unit.unique():
        window_size = cfg[0]
        collection = []
        temp_df = df[df.unit == unit]
        temp_df_values = temp_df.values
        EoL = temp_df.cycles.max()
        for _ in range(len(temp_df)):
            if window_size < EoL:
                collection.append(temp_df_values[:window_size, ])  # starting from 0 every time!
                window_size += cfg[0]
            elif window_size == EoL:
                collection.append(temp_df_values)
                break
            else:
                collection.append(temp_df_values)
                break
        subsequences[unit] = collection

    # creating the dataframe of expanded windows
    final = []
    lookup = {}
    prev = 0
    train_index = 0
    summ = 0
    for i in df.unit.unique():
        subs = subsequences[i]
        summ += len(subs)
        for j in range(1, len(subs) + 1):
            temp_df = pd.DataFrame(np.array(subs[j - 1]))
            temp_df[0] = np.repeat(prev + j, len(temp_df))
            final.append(temp_df)
            lookup[prev + j] = [train_index, train_index + len(subs[j - 1])]
        prev += j
        train_index += len(subs[j - 1])

    df_final = pd.concat(final)
    df_final.columns = cols
    df_final = df_final.astype({'cycles': 'int64'})

    print(f'df_final shape is: {df_final.shape}')

    # extracting features from every expanding window
    X = extract_features(df_final, column_id="unit", column_sort="cycles", n_jobs=n_jobs,
                         default_fc_parameters=EfficientFCParameters())
    impute(X)

    # creating the target RUL
    target_RUL = []
    for i in df.unit.unique():
        D = df[df.unit == i]
        subs = subsequences[i]
        EOL = len(D)
        init_RUL = cfg[1]
        for j in range(len(subs)):
            temp_df = pd.DataFrame(np.array(subs[j]))
            if len(temp_df) <= EOL / 2:
                target_RUL.append(init_RUL)

            else:
                if EOL % cfg[0] == 0:
                    s = EOL / 2
                else:
                    s = EOL / 2 - (EOL / 2) % cfg[0]
                p = (0 - init_RUL) / (EOL - s)
                rul = p * len(temp_df) - p * EOL
                target_RUL.append(rul)

    Y = pd.DataFrame(target_RUL, columns=['RUL'], index=range(1, len(target_RUL) + 1))

    extracted_features_test = extract_features(test, column_id="unit", column_sort="cycles", n_jobs=n_jobs,
                                               default_fc_parameters=EfficientFCParameters())
    impute(extracted_features_test)

    return X, Y, subsequences, extracted_features_test, RUL


# timeliness
def timeliness(rul, pred_rul):
    """
    The function calculates the timeliness score.
    See here for more information: https://doi.org/10.1109/PHM.2008.4711414

    :param rul: The ground truth RUL
    :param pred_rul: The estimated RUL
    :return: The timeliness score
    """
    tmls = []
    if isinstance(rul, pd.DataFrame) or isinstance(rul, pd.Series):
        rul = rul.values
    for i in range(len(rul)):
        dif = pred_rul[i] - rul[i]  # this was mistakenly as rul[i] - pred_rul[i]
        if dif < 0:
            tmls.append(np.exp((1 / 13) * np.abs(dif)) - 1)
        else:
            tmls.append(np.exp((1 / 10) * np.abs(dif)) - 1)
    S = np.sum(tmls)
    return S


# RMSE
def RMSE(rul, pred_rul):
    """
    This function calculates the RMSE score.

    :param rul: The ground truth RUL
    :param pred_rul: The estimated RUL
    :return: The timeliness score
    """
    if isinstance(rul, pd.DataFrame) or isinstance(rul, pd.Series):
        rul = rul.values

    return np.sqrt(mean_squared_error(rul, pred_rul))


if __name__ == "__main__":
    rep = sys.argv[1]
    n_jobs = multiprocessing.cpu_count()

    output_file = 'TIM_RUN-local-' + str(rep)
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    for t, rul in zip([1, 2, 3, 4], [115, 135, 125, 135]):
        start = time.time()
        start_cpu = time.clock()

        date = time.strftime("%Y%m%d_%H%M%S")

        print('Corrected timeliness')
        print(f'--- Started Pipeline on dataset {t} ({date}) ---')

        cfg = [10, rul]

        print('Pre-processing with window size: ' + str(cfg[0]) + ' and init_rul: ' + str(cfg[1]))

        X, y, subsequences, extracted_features_test, RUL = preprocessing(cfg, t, n_jobs)
        print(f"Shape of X is: {X.shape}")

        print('Feature selection')

        X = select_features(X, y['RUL'], ml_task='regression', n_jobs=n_jobs)  # or boruta

        print(f"Shape of X after selection is: {X.shape}")

        extracted_features_test = extracted_features_test[X.columns]

        timeliness_score = make_scorer(timeliness, greater_is_better=False)
        rmse_score = make_scorer(RMSE, greater_is_better=False)

        print('Started TPOT')

        GENERATIONS = 10
        POPULATION = 20
        SEED = eval(rep)
        MAX_EVAL = 5

        print(f'The SEED is {SEED}')

        with open('./' + str(output_file) + '/' + str(date) + 'train_features_cmapss_piece_wise_linear_dataset_' + str(
                t) + '_' + str(GENERATIONS) + '_' + str(POPULATION) + '_' + str(SEED) + '_' + str(MAX_EVAL) + '.pkl',
                  'wb') as f:
            pkl.dump(list(X.columns), f)

        tpot = TPOTRegressor(generations=GENERATIONS, population_size=POPULATION, verbosity=3, n_jobs=n_jobs,
                             scoring=timeliness_score, max_eval_time_mins=MAX_EVAL, random_state=SEED)

        tpot.fit(X, y)

        print('Finished fitting')

        print('Inferring..')
        predictions = tpot.predict(extracted_features_test)

        predictions = predictions.reshape(len(RUL), 1)

        score = timeliness(RUL, predictions)
        rmse = np.sqrt(mean_squared_error(RUL, predictions))

        print('--- The score on the test set is: ' + str(score) + ' and rmse is:' + str(rmse) + ' ---')

        end = time.time()
        end_cpu = time.clock()

        print('--- Finished ---')
        print('--- Elapsed time: ' + str((end - start) / 60) + ' minutes')
        print('--- Elapsed cpu time: ' + str((end_cpu - start_cpu) / 60) + ' minutes')

        tpot.export(
            './' + str(output_file) + '/' + str(date) + 'tpot_cmapss_piece_wise_linear_dataset_' + str(t) + '_' + str(
                GENERATIONS) + '_' + str(POPULATION) + '_' + str(SEED) + '_' + str(MAX_EVAL) + '.py')
