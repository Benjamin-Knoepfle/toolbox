# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:56:13 2015

@author: root
"""


PATH = '/home/borschi/data_science/how_much_did_it_rain/data/'
RAW_PATH = PATH + 'raw_data/'
WORK_PATH = PATH + 'work_data/'
FINAL_PATH = PATH + 'final_data/'

GLOBAL_TRAIN = RAW_PATH+'train_2013.csv'
GLOBAL_TEST = RAW_PATH+''
PREFIX = 'test'
LOKAL_TRAIN = WORK_PATH+PREFIX+'_train.csv'
LOKAL_TEST = WORK_PATH+PREFIX+'_test.csv'
SUBMISSION = FINAL_PATH+PREFIX+'_submission.csv'

ID_FIELD = ''
TARGET_FIELD = ''

NUMBER_OF_CORS= 4
TRAIN_RATIO = 0.8


