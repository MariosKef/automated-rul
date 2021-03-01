# Automated Machine Learning for Remaining Useful Life Estimation of Aircraft Engines

![MIT License](https://img.shields.io/github/license/MariosKef/automated_rul?style=plastic)

## Introduction

This repository holds the source code used for the work on the paper *Automated Machine Learning for Remaining Useful Life Estimation of Aircraft Engines*
(currently under review).

The work focuses on the difficulty of the estimation of the remaining useful life (RUL), emphasizes on the importance of careful pre-processing of the raw data
and recommends the usage of AutoML to perform the estimation task, by suggesting optimal pipelines. The proposed methodology is validated on the widely used CMAPSS
dataset.

This repository is the work of Marios Kefalas, PhD candidate at the Leiden Institute of Advanced Computer Science (LIACS), Leiden University, Leiden, The Netherlands.

## Usage
To run the experiments you can do the following:
	* clone the repository
	* cd to the directory of the cloned repository
	* For the proposed methodology run the *rul_pipeline.py* as
		- *python<version_number> rul_pipeline.py a*, where a is an integer that can be used to track experiments and is used as to set the random seed for reproducibility.
	* For the baseline run the *rul_baseline.py a*, where a is an integer that can be used to track experiments and is used as to set the random seed for reproducibility.

**Note:** To reproduce the work in the paper use the following seeds:
* For the proposed methodology: 2016, 2013, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
* For the baseline: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

## Acknowledgements 
This work is part of the research programme Smart Industry SI2016 with project name CIMPLO and project number 15465, which is partly financed by the Netherlands Organisation for Scientific Research (NWO).
