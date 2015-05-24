# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:23:52 2015

@author: root
"""

import sys
import logging

import settings
import feature_engineering
from ml_model import ML_Model
import submission_writer

# configure logging
logger = logging.getLogger("submission_pipeline")

handler = logging.FileHandler(settings.LOG_PATH)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# Feature engineer ( Transform features and create new ones )
logger.info('Start Feature Engineering')

test_data = feature_engineering.engineer_features( settings.GLOBAL_TEST, settings.PROCESSED_GLOBAL )

#test_data.drop( settings.ID_FIELD, inplace=True, axis=1 )

# Train model and evaluate 
logger.info('Train and Evaluate Model')
model = ML_Model()
model.load_model()
ids, prediction = model.predict_submission()

submission_writer.create_submission( ids, prediction )