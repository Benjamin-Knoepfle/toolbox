# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:00:49 2015

@author: root
"""
import sys
import logging

import settings
import initial_preparation
import feature_engineering
from ml_model import ML_Model

# configure logging
logger = logging.getLogger("train_pipeline")

handler = logging.FileHandler(settings.LOG_PATH)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# Create a "local" train and testset
initial_preparation.create_test_and_train_set()

# Feature engineer ( Transform features and create new ones )
logger.info('Start Feature Engineering')
fe = feature_engineering.FeatureEngineer()
train_data = fe.engineer_features(settings.LOKAL_TRAIN, settings.PROCESSED_TRAIN)
fe.train_state = False
test_data = fe.engineer_features (settings.LOKAL_TEST, settings.PROCESSED_TEST )

# Train model and evaluate 
logger.info('Train and Evaluate Model')
model = ML_Model()
score = model.train_model()
model.store_model()
print(score)

    