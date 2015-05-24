# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:19:14 2015

@author: root
"""

import numpy as np

def rmsle( prediction, actuals ):
    prediction = np.where(prediction<0,0,prediction)
    normalizer = 1.0/len(prediction)
    squared_log_error = ( np.log(prediction +1) - np.log(actuals +1) )**2
    result = ( normalizer * squared_log_error.sum() )**0.5
    return result

def dummy( prediction, actuals ):
    prediction = np.where(prediction<0,0,prediction)
    normalizer = 1.0/len(prediction)
    squared_log_error = (prediction - actuals  )**2
    result = ( normalizer * squared_log_error.sum() )**0.5
    return result
