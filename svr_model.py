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
        
        
    def fit( self, features, targets, cv):
        if cv:
            self.clf = grid_search.GridSearchCV( svm.SVR(), 
                                                self.paramgrid, 
                                                verbose=5,
                                                scoring=make_scorer( dummy,
                                                                    greater_is_better=False
                                                                    )
                                                )
            self.clf.fit( features, targets )
            self.clf = self.clf.best_estimator_        
        else:
            self.clf = svm.SVR()
            self.clf.fit( features, targets )
        
    def predict( self, features ):
        return self.clf.predict( features )
        
     