# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:35:53 2015

@author: root
"""

import pandas as pd
from sklearn.externals import joblib
from sklearn import grid_search
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

import settings

class ML_Model():

    def __init__( self, settings ):
        self.settings = settings


    def store_model( self ):
        joblib.dump(self.clf, self.settings.model_path) 
        
    def load_model(self):
        self.clf = joblib.load( self.settings.model_path )


    def train( self ):
        train_features, train_targets = self.get_datasets_( self.settings.processed_train_data_path )
        self.gridsearch( train_features, train_targets)
        test_features, test_targets = self.get_datasets_( self.settings.processed_test_data_path )
        score = self.evaluate_model_( test_features, test_targets )
        features, targets = self.get_datasets_( self.settings.processed_data_path )
        self.train_model_( features, targets )
        self.store_model()
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
      
    def gridsearch( self,  features, targets ):
        self.classy = grid_search.GridSearchCV( self.clf, 
                                           self.paramgrid, 
                                           n_jobs = settings.NUMBER_OF_CORS,
                                           verbose=5
                                         )
        self.classy.fit( features, targets )
        self.clf = self.classy.best_estimator_ 
        
        
    def train_model_( self, train_features, train_targets ):
        self.clf.fit( train_features, train_targets )
        
        
    def evaluate_model_( self, test_features, test_targets ):
        prediction = self.clf.predict( test_features )
        score = settings.score(test_targets, prediction)
        self.test_targets = test_targets
        self.predicted_targets = prediction
        return score
        
    
    def predict(self):
        self.load_model()
        features, ids = self.get_datasets_( self.settings.processed_submission_data_path, submission=True)
        prediction = self.clf.predict( features )
        pd.DataFrame({'id':id, 'prediction':prediction}).to_csv( self.settings.predicted_data_path, sep=';', index=False)
        return ids, prediction
        
        

            
        
class Regression_Model( ML_Model ):

    def evaluate_model_( self, test_features, test_targets ):
        prediction = self.clf.predict( test_features )
        score = settings.score(test_targets, prediction)
        self.test_targets = test_targets
        self.predicted_targets = prediction
        return score
 

           
class Classification_Model( ML_Model ):
    
    def evaluate_model_( self, test_features, test_targets ):
        prediction = self.clf.predict( test_features )
        score = settings.score(test_targets, prediction)
        self.test_targets = test_targets
        self.predicted_targets = prediction
        return score
        
        
    def plot_confusion_matrix( self ):
        cm = confusion_matrix(self.test_targets, self.predicted_targets)
        # Show confusion matrix in a separate window
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


    def plot_roc( self ):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(1, self.n_classes+1):
            fpr[i], tpr[i], _ = roc_curve(self.test_targets == i, self.predicted_target_prob[:,i-1])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(1, self.n_classes+1):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.show()        
        
        
    