# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:18:45 2015

@author: root
"""


import sys
import logging

import pandas as pd
import numpy as np

import settings

# configure logging
logger = logging.getLogger("feature engineering")

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def explore_features( data ):
    logger.info('start explore_features')
    return data
    
def treat_features( data ):
    logger.info('start treat_features')
    return data
    
def get_mean_value( x ):
    array = np.array( x.split(' '), dtype='float' )
    mean_ = array.mean()
    return np.where(mean_<0, 0, mean_)
    
def transform_features( data ):
    logger.info('start transform_features')
    data['RR1_mean'] = data['RR1'].map( get_mean_value )
    return data
    
def create_features( data ):
    logger.info('start create_features')
    return data
    
def select_features( data ):
    droplist = ['TimeToEnd', 'DistanceToRadar', 'Composite', 'HybridScan', 'HydrometeorType', 
            'Kdp', 'RR1', 'RR2', 'RR3', 'RadarQualityIndex', 'Reflectivity', 'ReflectivityQC', 
            'RhoHV', 'Velocity', 'Zdr', 'LogWaterVolume', 'MassWeightedMean', 'MassWeightedSD']
    logger.info('start select_features')
    return data.drop( droplist, axis=1 )
    
def engineer_features( data ):
    logger.info('start engineer_features')
    data = treat_features( data )
    data = transform_features( data )
    data = create_features( data )
    data = select_features( data )
    return data


if __name__ == '__main__':
    logger.info('Sorry man')