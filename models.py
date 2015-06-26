# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:05:54 2015

@author: bknoepfl
"""
from sklearn import svm
from sklearn import tree
from sklearn import ensemble

from settings import Settings
import ml_model


class SVR( ml_model.Regression_Model ):
    
    def __init__( self, settings ):
        super(SVR, self).__init__( settings )
        self.paramgrid = {'C':[0.1, 0.5, 1, 5, 10], 'degree':[2,3,4,5]}
        self.clf = svm.SVR()
    
    
class Decision_Tree_Regressor( ml_model.Regression_Model ):
    
    def __init__( self, settings ):
        super(Decision_Tree_Regressor, self).__init__( settings )
        self.paramgrid = {'max_depth':[2, 5, 10]}
        self.clf = tree.DecisionTreeRegressor()
 
class Extra_Trees_Regressor( ml_model.Regression_Model ): 
    def __init__( self, settings ):
        super(Extra_Trees_Regressor, self).__init__( settings )
        self.paramgrid = {'n_estimators':[5, 10, 25]}
        self.clf = ensemble.ExtraTreesRegressor()
 
class Model():
    
    def get_model( self, settings ):
        if settings.model == 'SVR':
            model = SVR( settings )
        elif settings.model == 'Decision_Tree_Regressor':
            model = Decision_Tree_Regressor( settings )
        else:
            settings.model = 'Extra_Trees_Regressor'
            settings.update()
            model = Extra_Trees_Regressor( settings )
        return model
 
        
        
        
if __name__ == '__main__':
    # Train model and evaluate 
    #logger.info('Train and Evaluate Model')
    model = Model().get_model( Settings() )
    score = model.train()
    model.predict()
    #print(score)