# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:40:54 2015

@author: root
"""

from sklearn import svm
from sklearn import grid_search
from sklearn.metrics import make_scorer
from my_metrics import dummy

import settings

class SVR():
    
    def __init__( self ):
        self.paramgrid = {'kernel':('linear', 'rbf', 'sigmoid'), 'C':[0.1, 0.5, 1, 5, 10], 'degree':[2,3,4,5]}
        self.clf = svm.SVR()
        
    def fit( self, features, targets):
        self.clf.fit( features, targets )
        
    def predict( self, features ):
        return self.clf.predict( features )
        
     