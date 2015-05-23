# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:56:13 2015

@author: root
"""

import sklearn.metrics 

import my_metrics 

PATH = '/home/borschi/data_science/bike_sharing/data/'
RAW_PATH = PATH + 'raw_data/'
WORK_PATH = PATH + 'work_data/'
FINAL_PATH = PATH + 'final_data/'
MODEL_PATH = PATH + 'model_data/'

GLOBAL_TRAIN = RAW_PATH+'train.csv'
GLOBAL_TEST = RAW_PATH+'test.csv'
PREFIX = 'baseline'
LOKAL_TRAIN = WORK_PATH+PREFIX+'_train.csv'
LOKAL_TEST = WORK_PATH+PREFIX+'_test.csv'
PROCESSED_TRAIN = WORK_PATH+PREFIX+'_processed_train.csv'
PROCESSED_TEST = WORK_PATH+PREFIX+'_processed_test.csv'
PROCESSED_GLOBAL = WORK_PATH+PREFIX+'_processed_global.csv'
SUBMISSION = FINAL_PATH+PREFIX+'_submission.csv'

MODEL = MODEL_PATH + PREFIX + 'model.pkl'

ID_FIELD = 'datetime'
TARGET_FIELD = 'count'

NUMBER_OF_CORS= 4
TRAIN_RATIO = 0.8

score = my_metrics.rmsle