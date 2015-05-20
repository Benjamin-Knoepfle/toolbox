# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:00:49 2015

@author: root
"""
import sys
import logging

import pandas as pd

import settings
import initial_preparation
import feature_engineering
from ml_model import ML_Model

# configure logging
logger = logging.getLogger("train_pipeline")

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# Create a "local" train and testset
initial_preparation.create_test_and_train_set()

# Feature engineer ( Transform features and create new ones )
logger.info('Start Feature Engineering')
train_data = pd.read_csv( settings.LOKAL_TRAIN, delimiter=',' )
train_data = feature_engineering.engineer_features( train_data )

train_data.drop( settings.ID_FIELD, inplace=True, axis=1 )
train_features = train_data.drop( settings.TARGET_FIELD, axis=1 )
train_targets = train_data[settings.TARGET_FIELD]

test_data = pd.read_csv( settings.LOKAL_TEST, delimiter=',' )
test_data = feature_engineering.engineer_features( test_data )

test_data.drop( settings.ID_FIELD, inplace=True, axis=1 )
test_features = train_data.drop( settings.TARGET_FIELD, axis=1 )
test_targets = train_data[settings.TARGET_FIELD]

# Train model and evaluate 
logger.info('Train and Evaluate Model')
model = ML_Model()
model.train_model( train_features, train_targets )
score = model.evaluate_model( test_features, test_targets )
print(score)