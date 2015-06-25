# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:05:54 2015

@author: bknoepfl
"""
from sklearn import svm
from sklearn import tree

import ml_model


class SVR( ml_model.Regression_Model ):
    
    def __init__( self ):
        self.paramgrid = {'kernel':('linear', 'rbf', 'sigmoid'), 'C':[0.1, 0.5, 1, 5, 10], 'degree':[2,3,4,5]}
        self.clf = svm.SVR()
        
class Decision_Tree_Regressor( ml_model.Regression_Model ):
    
    def __init__( self ):
        self.paramgrid = {'max_depth':[2, 5, 10]}
        self.clf = tree.DecisionTreeRegressor()
 