# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:16:00 2015

@author: root
"""
import sys
import logging


import matplotlib.pyplot as plt
import pandas as pd

import settings


# configure logging
logger = logging.getLogger("feature_exploration")

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


numerical_features = ['temp', 'atemp', 'humidity', 'windspeed']
categorical_features = ['season', 'holiday', 'workingday', 'weather']
time_feature = ['datetime']
target = 'count'



def create_histogram( data, feature ):
    plt.figure()
    plt.title( feature )
    plt.hist( data[feature] )
    plt.savefig( settings.EXPO_FIGS+feature+'_hist.png' )
    
def create_boxplot( data, feature ):
    plt.figure()
    plt.title( feature )
    plt.boxplot( data[feature], vert=False )    
    plt.savefig( settings.EXPO_FIGS+feature+'_box.png' )    
    
    
def create_scatter( data, feature_1, feature_2 ):
    plt.figure()
    plt.title( feature_1+'_'+feature_2 )
    plt.xlabel( feature_1 )
    plt.ylabel( feature_2 )
    plt.scatter( data[feature_1], data[feature_2] )
    plt.savefig( settings.EXPO_FIGS+feature_1+'_'+feature_2+'_scatter.png' )

def create_statistics( data, feature ):
    create_univarite_analysis()
    create_bivariate_analysis()


def create_univarite_analysis():
    create_numerical_analysis()
    create_categorical_analysis()


def create_numerical_analysis( data ):
    logger.log( data[ numerical_features ].describe() )
    for num in numerical_features:
        create_histogram( data, num )
        create_boxplot( data, num )


def create_categorical_analysis( data ):
    for cat in categorical_features:
        tmp = [cat, target]
        #if settings.classification
        #data.groupby(tmp)[target].count().unstack().plot(kind='bar', stacked=True)
        #else
        data.groupby([target]).boxplot()

def create_bivariate_analysis():
    create_feature_feature_analysis()
    create_feature_targets_analysis()
    
    
if __name__ == '__main__':
    data = pd.read_csv( settings.LOKAL_TRAIN, delimiter=',' )
    create_univarite_analysis()
    create_bivariate_analysis()
