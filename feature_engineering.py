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

    def __init__(self, train=True):
        self.numerical_features = ['temp', 'atemp', 'humidity', 'windspeed']
        self.categorical_features = ['season', 'holiday', 'workingday', 'weather']
        self.time_features = []
        self.text_features = []
        
        self.train_state = train
    
        self.standard_scaler = {}    
    
    
    def store_model(self):
        joblib.dump(self.standard_scaler, settings.FEATURE+ 'standard_feature.pkl' ) 
        
    def load_model(self):
        self.standard_scaler = joblib.load( settings.FEATURE+ 'standard_feature.pkl' )
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
        
    def transform_numerical_features( self, data ):        
        for numerical in self.numerical_features:
            feat_name = numerical+'_z'
            if self.train_state:
                self.standard_scaler[feat_name] = preprocessing.StandardScaler()
                data[feat_name] = self.standard_scaler[feat_name].fit_transform( data[numerical] )
            else:
                data[feat_name] = self.standard_scaler[feat_name].transform( data[numerical] )
        return data
        
    def transform_features( self, data ):
        data = self.transform_numerical_features( data )
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
        
    def engineer_features( self, infile, outfile ):
        logger.info('start engineer_features')
        if not self.train_state:
            self.load_model()
        data = pd.read_csv( infile, delimiter=',' )
        data = self.treat_features( data )
        data = self.transform_features( data )
        data = self.create_features( data )
        data = self.select_features( data )
        data.to_csv( outfile, sep=',', index=False )
        if self.train_state:
            self.store_model()
        return data


if __name__ == '__main__':
    fe = FeatureEngineer( False )
    data = fe.engineer_features(settings.LOKAL_TRAIN, settings.PROCESSED_TRAIN)