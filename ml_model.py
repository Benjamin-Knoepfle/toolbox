# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:35:53 2015

@author: root
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib

import settings
import svr_model

class ML_Model():

    def __init__(self):
        self.clf = svr_model.SVR()

    def store_model(self):
        joblib.dump(self.clf, settings.MODEL) 
        
    def load_model(self):
        self.clf = joblib.load( settings.MODEL )


    def train_model( self ):
        train_features, train_targets = self.get_datasets_( settings.PROCESSED_TRAIN )
        self.train_model_( train_features, train_targets, cv=True )
        test_features, test_targets = self.get_datasets_( settings.PROCESSED_TEST )
        train_features = train_features.append( test_features )
        train_targets = train_targets.append( test_targets )
        score = self.evaluate_model_( test_features, test_targets )
        self.train_model_( train_features, train_targets )
        return score

    def get_datasets_( self, infile, submission=False ):
        data = pd.read_csv( infile, delimiter=',' )
        ids = data[settings.ID_FIELD]
        data.drop( settings.ID_FIELD, inplace=True, axis=1 )
        if submission:
            return( data, ids )
        else:
            features = data.drop( [settings.TARGET_FIELD,'casual', 'registered'], axis=1 )  
            targets = data[settings.TARGET_FIELD]
            return (features,targets)        
        
        
    def train_model_( self, train_features, train_targets, cv=False):
        self.clf.fit( train_features, train_targets, cv )
        
        
    def evaluate_model_( self, test_features, test_targets ):
        prediction = self.clf.predict( test_features )
        score = settings.score(test_targets, prediction)
        return score
        
    
    def predict_submission(self):
        features, ids = self.get_datasets_( settings.PROCESSED_GLOBAL, submission=True)
        prediction = self.clf.predict( features )
        return ids, prediction