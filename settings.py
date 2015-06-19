# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:56:13 2015

@author: root
"""

import datetime

import sklearn.metrics as metrics
import my_metrics 
import svr_model as ml_model

TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M")

PATH = '/opt/bknoepfle/data_science/bike_sharing/'

RAW_PATH = PATH + 'data/raw_data/'
WORK_PATH = PATH + 'data/work_data/'
FINAL_PATH = PATH + 'data/final_data/'
MODEL_PATH = PATH + 'data/model_data/'

FEATURE_PREFIX = 'with_date'
MODEL_PREFIX = '24_5_svr'

EXPO_FIGS = PATH + 'figures/exploration_figures/'+TIME+'_'
LOG_PATH = PATH + 'logs/'+TIME+'_'+FEATURE_PREFIX+'_'+MODEL_PREFIX+'.log'

GLOBAL_TRAIN = RAW_PATH+'train.csv'
GLOBAL_TEST = RAW_PATH+'test.csv'

LOKAL_TRAIN = WORK_PATH+'train.csv'
LOKAL_TEST = WORK_PATH+'test.csv'

PROCESSED_TRAIN = WORK_PATH+FEATURE_PREFIX+'_processed_train.csv'
PROCESSED_TEST = WORK_PATH+FEATURE_PREFIX+'_processed_test.csv'
PROCESSED_GLOBAL = WORK_PATH+FEATURE_PREFIX+'_processed_global.csv'

SUBMISSION = FINAL_PATH+MODEL_PREFIX+'_submission.csv'

MODEL = MODEL_PATH + MODEL_PREFIX + '_model.pkl'
FEATURE = MODEL_PATH + FEATURE_PREFIX 

model = ml_model.Regression_Model()

ID_FIELD = 'datetime'
TARGET_FIELD = 'count'

NUMBER_OF_CORS= 4
TRAIN_RATIO = 0.8

score = metrics.mean_squared_error