
# -*- coding: utf-8 -*-

"""

Created on Tue Mar 03 16:45:41 2015


@author: bknoepfl

"""


import numpy as np
from sklearn import grid_search
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt



n_cores = 2


class ML_Model:

    def make_grid_search( self, data, target, score=None ):
        self.clf = grid_search.GridSearchCV(self.clf, self.parameters, n_jobs=n_cores, verbose=2)
        self.clf.fit( data, target)
        self.n_classes = len( np.unique( target ) )



    def evaluate_model( self, test_data, test_target, score=accuracy_score ):
        self.test_target = test_target
        self.predicted_target_prob = self.clf.predict_proba( test_data )
        self.predicted_target = self.clf.predict( test_data )


    def plot_confusion_matrix( self ):
        cm = confusion_matrix(self.test_target, self.predicted_target)
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
            fpr[i], tpr[i], _ = roc_curve(self.test_target == i, self.predicted_target_prob[:,i-1])
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
    



if __name__ == '__main__':
    rfm = RandomForestModel()
    rfm.make_grid_search( data_train, target_train )
    rfm.evaluate_model( data_test, target_test )
    rfm.plot_roc()


