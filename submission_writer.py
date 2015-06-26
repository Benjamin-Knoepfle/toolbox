# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:29:12 2015

@author: root
"""
import logging

import numpy as np
import pandas as pd

import settings
from settings import Settings

class IO_Manager():
    
    def __init__( self, settings ):
                    
        # configure loggin
        self.logger = logging.getLogger("IO_Manager")
        handler = logging.FileHandler(settings.LOG_PATH)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s: %(message)s'))
        
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self.settings = settings
    
    def write_submission_file( self ):
        prediction = pd.read_csv( self.settings.predicted_data_path, delimiter=';' )
        prediction[ prediction['prediction']<0 ] = 0  
        prediction.columns = ['datetime','count']
        prediction.to_csv( self.settings.submission_data_path, index=False )


class Single_Source_Reader( IO_Manager ):
    
    def __init__( self, settings ):
        super(Single_Source_Reader, self).__init__( settings )
    
    
    def prepare_data( self, raw_data ):
        self.logger.info('start prepare_data')    
        # create train and test-set
        sampler = np.random.random( raw_data.shape[0] ) < settings.TRAIN_RATIO
        
        train_data = raw_data[sampler]
        test_data = raw_data[-sampler]
        
        self.logger.info('finished prepare_data')
        return( train_data, test_data )

    def create_test_and_train_set( self ):
        # Read Kaggle train data
        self.logger.info('Reading global train set')
        raw_data = pd.read_csv( settings.GLOBAL_TRAIN, delimiter=',')
        
        self.logger.info('Create local train and test sets')
        (train_data, test_data) = self.prepare_data( raw_data )
        
        self.logger.info('write out train and test set')
        train_data.to_csv( settings.LOKAL_TRAIN, index=False )
        test_data.to_csv( settings.LOKAL_TEST, index=False )
        
        
    
if __name__ == '__main__':
    io = IO_Manager( Settings() )
    io.write_submission_file()