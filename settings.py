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


EXPO_FIGS = PATH + 'figures/exploration_figures/'+TIME+'_'
LOG_PATH = PATH + 'logs/'+TIME+'_'

GLOBAL_TRAIN = RAW_PATH+'train.csv'
GLOBAL_TEST = RAW_PATH+'test.csv'

LOKAL_TRAIN = WORK_PATH+'train.csv'
LOKAL_TEST = WORK_PATH+'test.csv'





ID_FIELD = 'datetime'
TARGET_FIELD = 'count'

NUMBER_OF_CORS= 4
TRAIN_RATIO = 0.8

score = metrics.mean_squared_error

Problem = 'Regression'

class Settings():
    
    def __init__( self ):
        self.io = 'single_source'
        self.feature_engineer = 'default_fe'
        self.model = 'ExtraTreesRegressor'        
                        
        self.description = 'initial pipeline with default settings'
        self.update()        
        
    def update( self ):
        self.pipeline = self.feature_engineer +'_'+ self.model
        self.logpath = self.pipeline+'.log'
        self.feature_engineer_path = MODEL_PATH + self.feature_engineer +'.pkl'
        self.model_path = MODEL_PATH + self.pipeline
 
        self.processed_train_data_path = WORK_PATH+self.feature_engineer+'_processed_train.csv'
        self.processed_test_data_path = WORK_PATH+self.feature_engineer+'_processed_test.csv'
        self.processed_data_path = WORK_PATH+self.feature_engineer+'_processed.csv'
        self.processed_submission_data_path = WORK_PATH+self.feature_engineer+'_processed_submission.csv'
        self.predicted_data_path = WORK_PATH+self.pipeline+'_prediction.csv'     
        self.submission_data_path = FINAL_PATH+self.pipeline+'_submission.csv'        
        
        
    def write( self, file_path ):
        with open( file_path+'settings.json', 'w' ) as settings_json:
            json.dump( self.__dict__, settings_json )
        
    def read( self, file_path ):
        with open( file_path+'settings.json') as settings_json:
            self.__dict__ = json.load( settings_json )