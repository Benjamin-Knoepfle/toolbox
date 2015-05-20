# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:03:34 2015

@author: root
"""

import sys
import logging

import numpy as np
import pandas as pd

import settings

# configure logging
logger = logging.getLogger("train_pipeline")

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def prepare_data( raw_data ):
    logger.info('start prepare_data')    
    # create train and test-set
    sampler = np.random.random( raw_data.shape[0] ) < settings.TRAIN_RATIO
    
    train_data = raw_data[sampler]
    test_data = raw_data[-sampler]
    
    logger.info('finished prepare_data')
    return( train_data, test_data )

def create_test_and_train_set():
    # Read Kaggle train data
    logger.info('Reading global train set')
    raw_data = pd.read_csv(settings.GLOBAL_TRAIN, delimiter=',')
    
    logger.info('Create local train and test sets')
    (train_data, test_data) = prepare_data( raw_data )
    
    logger.info('write out train and test set')
    train_data.to_csv( settings.LOKAL_TRAIN, index=False )
    test_data.to_csv( settings.LOKAL_TEST, index=False )