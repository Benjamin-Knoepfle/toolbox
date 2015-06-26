# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:18:45 2015

@author: root
"""


import sys
import logging

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib

import settings
import feature_exploration 
import feature_transformation

# configure logging
logger = logging.getLogger("feature engineering")

handler = logging.FileHandler(settings.LOG_PATH)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class FeatureEngineer():

    def __init__(self, settings):
        self.settings = settings
        self.numerical_features = ['temp', 'atemp', 'humidity', 'windspeed']
        self.categorical_features = ['season', 'holiday', 'workingday', 'weather']
        self.time_features = []
        self.text_features = []
            
        self.standard_scaler = {}    
    
    
    def store_model(self):
        joblib.dump(self.standard_scaler, self.settings.feature_engineer_path ) 
        
    def load_model(self):
        self.standard_scaler = joblib.load( self.settings.feature_engineer_path )
    '''
    Hieraus muss ich noch ein objekt machen, das zustandsabhangige transformationen speicher kann.
    Wie zb. scaler = preprocessing.StandardScaler().fit(X) und pca, etc
    '''

    def explore_features( self, data ):
        logger.info('start explore_features')
        ## univariate and bivariate feature analysis
        ## create index table
        ## random forest method
        ## clustering
        return data
        
    def treat_features( self, data ):
        logger.info('start treat_features')
        return data
        
    def get_mean_value( self, x ):
        array = np.array( x.split(' '), dtype='float' )
        mean_ = array.mean()
        return np.where(mean_<0, 0, mean_)
        
    def transform_numerical_features( self, data, train_state=True ):        
        for numerical in self.numerical_features:
            feat_name = numerical+'_z'
            if train_state:
                self.standard_scaler[feat_name] = preprocessing.StandardScaler()
                data[feat_name] = self.standard_scaler[feat_name].fit_transform( data[numerical] )
            else:
                data[feat_name] = self.standard_scaler[feat_name].transform( data[numerical] )
        return data
        
    def transform_features( self, data, train_state=True ):
        data = self.transform_numerical_features( data, train_state )
        return data
            
        
    def create_features( self, data ):
        logger.info('start create_features')
        #data = pd.get_dummies( data, ['season', 'weather'] ) 
        data['hours'] = data['datetime'].apply( self.get_hour )   
        data['day'] = data['datetime'].apply( self.get_day )    
        data['month'] = data['datetime'].apply( self.get_month )        
        return data
    
    def get_hour(self, x):
        return int(x[-8:-6])
    
    def get_day(self, x):
        return int( x[-11:-9])
    
    def get_month(self, x):
        return int(x[-14:-12])
    
        
    # important do not drop identifier or target ;)
    def select_features(self, data ):
        droplist = self.numerical_features
        logger.info('start select_features')
        return data.drop( droplist, axis=1 )
        
        
    def train( self ):
        logger.info('start train')
        # Train_data
        data = pd.read_csv( settings.LOKAL_TRAIN )
        train_data = self.fit_transform( data )  
        train_data.to_csv( self.settings.processed_train_data_path )
        # Test_data
        data = pd.read_csv( settings.LOKAL_TEST )
        test_data = self.transform( data )  
        test_data.to_csv( self.settings.processed_test_data_path )
        # All_data
        data = pd.read_csv( settings.GLOBAL_TRAIN )
        data = self.fit_transform( data )
        data.to_csv( self.settings.processed_data_path )
        # Store FE
        self.store_model()
        
    def predict( self ):
        logger.info('start predict')
        self.load_model()
        data = pd.read_csv( settings.GLOBAL_TEST )
        data = self.transform( data ) 
        data.to_csv( self.settings.processed_submission_data_path )
        
        
    def fit_transform( self, data ):
        logger.info('start fit_transform')
        data = self.treat_features( data )
        data = self.transform_features( data )
        data = self.create_features( data )
        data = self.select_features( data )
        return data
        
    def transform( self, data ):
        logger.info('start transform')
        data = self.treat_features( data )
        data = self.transform_features( data, False )
        data = self.create_features( data )
        data = self.select_features( data ) 
        return data
        
        



if __name__ == '__main__':
    fe = FeatureEngineer( settings.Settings() )
    #fe.train()
    fe.predict()