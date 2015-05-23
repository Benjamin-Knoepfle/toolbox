# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:29:12 2015

@author: root
"""

import settings

import numpy as np
import pandas as pd

def create_submission( ids, prediction ):
    prediction = np.where(prediction<0,0,prediction)
    submission_data = pd.DataFrame( {'datetime': ids, 'count':prediction} )
    submission_data.to_csv( settings.SUBMISSION, index=False )
    return submission_data
    