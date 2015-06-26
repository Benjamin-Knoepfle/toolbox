# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:00:49 2015

@author: root
"""
import sys
import logging


from sklearn.externals import joblib

import settings
from settings import Settings
import initial_preparation
from feature_engineering import FeatureEngineer
from models import Model
import submission_writer

class Pipeline():
    
    def __init__( self, feature_engineer=None, model=None ):
        # configure logging
        self.logger = logging.getLogger("Pipeline")
        handler = logging.FileHandler(settings.LOG_PATH)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s: %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        
        self.settings = Settings()        
        self.feature_engineer = FeatureEngineer( self.settings )
        self.model = Model().get_model( Settings() )
            
        
    def read( self, file_path ): 
        self.settings.read( file_path )
        self.read_feature_engineer( settings.feature_engineer_path )
        self.read_model( settings.model_path )
        
    def read_feature_engineer( self, file_path ):
        print('implement this')
        
    def read_model( self, file_path ):
        self.model = joblib.load( file_path+'_main.pkl' ) 
        self.model.load_model(file_path)          
        
        
    def write( self, file_path ):
        self.settings.write( file_path )
        self.write_feature_engineer( settings.feature_engineer_path )
        self.write_model( settings.model_path )
        
    def write_feature_engineer( self, file_path ):
        print('implement this')
        
    def write_model( self, file_path ):
        joblib.dump( self.model, file_path+'_main.pkl' ) 
        self.model.store_model(file_path)  
        
        
    def set_feature_engineer( self, feature_engineer ):
        self.feature_engineer = feature_engineer
        
    def set_model( self, model ):
        self.model = model


    def train( self ):
        # Create a "local" train and testset
        initial_preparation.create_test_and_train_set()

        # Feature engineer ( Transform features and create new ones )
        self.logger.info('Start Feature Engineering')
        train_data = self.feature_engineer.engineer_features(settings.LOKAL_TRAIN, settings.PROCESSED_TRAIN)
        self.feature_engineer.train_state = False
        test_data = self.feature_engineer.engineer_features (settings.LOKAL_TEST, settings.PROCESSED_TEST )

        # Train model and evaluate 
        self.logger.info('Train and Evaluate Model')
        score = self.model.train_model()
        self.model.store_model()
        self.logger.info('Model Score is %f', score)
        
    def predict( self ):
        # Feature engineer ( Transform features and create new ones )
        self.logger.info('Start Feature Engineering')
        self.feature_engineer.train_state = False
        test_data = self.feature_engineer.engineer_features( settings.GLOBAL_TEST, settings.PROCESSED_GLOBAL )
        
        #test_data.drop( settings.ID_FIELD, inplace=True, axis=1 )
        
        # Train model and evaluate 
        self.logger.info('Train and Evaluate Model')
        self.model.load_model()
        ids, prediction = self.model.predict_submission()
        
        submission_writer.create_submission( ids, prediction )


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.train()
    pipeline.predict()