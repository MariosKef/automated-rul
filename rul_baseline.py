import multiprocessing
import os
import sys
import time

import numpy as np
import pandas as pd
import tpot.config
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tpot import TPOTRegressor

del tpot.config.regressor_config_dict['xgboost.XGBRegressor']


def preprocessing(t):
    """
       This function takes loads raw sensor data
       and the ground truth RUL values of the test
       data standardizes them, performs the expanding
       window transformation over the training set,
       creates the RUL labels in a piece-wise linear
       manner and extracts statistics from the expanded
       windows of the training set and from the entirety
       of the test set.

       :param t: dataset number (1-4)
       :return:
           X: The statistical features of all expanded windows of the training set
           Y: The RUL of every sample in X
           extracted_features_test: The statistical features of all expanded windows of the test set
           RUL: The ground truth RUL values of the test set

       """

    train = pd.read_csv('./Data//Data_set' + str(t) + '/train_FD00' + str(t) + '.csv', parse_dates=False, delimiter=" ",
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

    # Here we check if t belongs to FD001 or FD003
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

    if t in [1, 3]:
        test = test_scaled.iloc[:, keep]
    else:
        test = test_scaled

    RUL = pd.read_csv('./Data/Data_set' + str(t) + '/RUL_FD00' + str(t) + '.csv', parse_dates=False, delimiter=" ",
                      decimal=".", header=None)
    RUL.drop(RUL.columns[[-1]], axis=1, inplace=True)
    RUL.index = range(1, len(RUL) + 1)

    cols = train.columns
    df = train

    print(f'Train shape: {train.shape}')
    print(f'Test shape: {test.shape}')
    print(f'RUL shape: {RUL.shape}')

    # creating the target RUL
    target_RUL = []
    target_per_unit = {}
    for i in df.unit.unique():
        temp_rul = []
        D = df[df.unit == i]
        EOL = len(D) - 1
        for j in range(len(D)):
            target_RUL.append(EOL - j)
            temp_rul.append(EOL - j)
        target_per_unit[i] = temp_rul

    Y = pd.DataFrame(target_RUL, columns=['RUL'], index=range(1, len(target_RUL) + 1))

    last_test = []
    for i in test.unit.unique():
        temp_test = test[test.unit == i]
        temp_test = temp_test.drop(['unit', 'cycles'], axis=1)
        last_test.append(temp_test.iloc[-1].values)

    last_test = np.array(last_test)

    return df, Y, last_test, RUL


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
    n_jobs = 64
    if n_jobs > 1:
        multiprocessing.set_start_method('forkserver')

    output_file = 'Baselines_run-' + str(rep)
    if not os.path.isdir(output_file):
        os.makedirs(output_file)

    for t in [1, 2, 3, 4]:
        start = time.time()
        start_cpu = time.clock()

        date = time.strftime("%Y%m%d_%H%M%S")

        print('Corrected timeliness')
        print(f'--- Started Pipeline on dataset {t} ({date}) ---')

        X, y, test, RUL = preprocessing(t)

        X.drop(['unit', 'cycles'], axis=1, inplace=True)

        print(f"Shape of X is: {X.shape}")

        timeliness_score = make_scorer(timeliness, greater_is_better=False)
        rmse_score = make_scorer(RMSE, greater_is_better=False)

        print('Started TPOT')

        GENERATIONS = 10
        POPULATION = 20
        SEED = eval(rep)
        MAX_EVAL = 5

        print(f'The SEED is {SEED}')

        tpot = TPOTRegressor(generations=GENERATIONS, population_size=POPULATION, verbosity=3, n_jobs=n_jobs,
                             scoring=timeliness_score, max_eval_time_mins=MAX_EVAL, random_state=SEED)

        tpot.fit(X, y.values.ravel())

        print('Finished fitting')

        print('Inferring..')
        predictions = tpot.predict(test)

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
