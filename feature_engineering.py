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

import settings
import feature_exploration 
import feature_transformation

# configure logging
logger = logging.getLogger("feature engineering")

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


numerical_features = ['temp', 'atemp', 'humidity', 'windspeed']
categorical_features = ['season', 'holiday', 'workingday', 'weather']


'''
Hieraus muss ich noch ein objekt machen, das zustandsabhangige transformationen speicher kann.
Wie zb. scaler = preprocessing.StandardScaler().fit(X) und pca, etc
'''


def explore_features( data ):
    logger.info('start explore_features')
    ## univariate and bivariate feature analysis
    ## create index table
    ## random forest method
    ## clustering
    return data
    
def treat_features( data ):
    logger.info('start treat_features')
    return data
    
def get_mean_value( x ):
    array = np.array( x.split(' '), dtype='float' )
    mean_ = array.mean()
    return np.where(mean_<0, 0, mean_)
    
def transform_features( data ):
    for numerical in numerical_features:
        feat_name = numerical+'_z'
        data[feat_name] = preprocessing.scale( data[numerical] )
    return data
        
    
def create_features( data ):
    logger.info('start create_features')
    #data = pd.get_dummies( data, ['season', 'weather'] ) 
    data['hours'] = data['datetime'].apply( get_hour )   
    data['day'] = data['datetime'].apply( get_day )    
    data['month'] = data['datetime'].apply( get_month )        
    return data

def get_hour(x):
    return x[-8:-6]

def get_day(x):
    return x[-11:-9]

def get_month(x):
    return x[-14:-12]

    
# important do not drop identifier or target ;)
def select_features( data ):
    droplist = numerical_features
    logger.info('start select_features')
    return data.drop( droplist, axis=1 )
    
def engineer_features( infile, outfile ):
    logger.info('start engineer_features')
    data = pd.read_csv( infile, delimiter=',' )
    data = treat_features( data )
    data = transform_features( data )
    data = create_features( data )
    data = select_features( data )
    data.to_csv( outfile, sep=',', index=False )
    return data


if __name__ == '__main__':
    data = engineer_features(settings.LOKAL_TRAIN, settings.PROCESSED_TRAIN)