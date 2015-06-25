# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:56:13 2015

@author: root
"""

import datetime
import json

import sklearn.metrics as metrics
import my_metrics 

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


ID_FIELD = 'datetime'
TARGET_FIELD = 'count'

NUMBER_OF_CORS= 4
TRAIN_RATIO = 0.8

score = metrics.mean_squared_error


class Settings():
    
    def __init__( self ):
        self.pipeline_name = 'default_pipeline'
        self.io = 'single_source'
        self.feature_engineer = 'default_fe'
        self.model = 'default_model'
        
        self.description = 'initial pipeline with default settings'
        
    def write( self, file_path ):
        with open( file_path+'settings.json', 'w' ) as settings_json:
            json.dump( self.__dict__, settings_json )
        
    def read( self, file_path ):
        with open( file_path+'settings.json') as settings_json:
            self.__dict__ = json.load( settings_json )