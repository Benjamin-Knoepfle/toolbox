# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:35:53 2015

@author: root
"""

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error

from crps import CRPS


class ML_Model():

    def __init__(self):
        self.clf = BayesianRidge()

    def train_model( self, train_features, train_targets):
        self.clf.fit( train_features, train_targets )
        
        
    def sigmoid(self, center, shape=0.5, length=70):
        # http://en.wikipedia.org/wiki/Sigmoid_function
        xs = np.arange(length)
        return 1. / (1 + np.exp(-(xs - center)/shape))   
        
    def crps_score( self, actual, prediction ):
        solution = []
        for idx, sample in enumerate(prediction):
            #result = [sample.Id]
            result =  self.sigmoid( sample ) 
            solution.append(result)
            if idx % 1000 == 0:
                print("Completed row %d" % idx)
        #solution = pd.DataFrame(solution, columns=solution_header)
        return CRPS( solution, actual )
        
    def evaluate_model( self, test_features, test_targets ):
        prediction = self.clf.predict( test_features )
        score = self.crps_score(test_targets, prediction)
        return score