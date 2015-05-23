# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:16:00 2015

@author: root
"""


import matplotlib.pyplot as plt
import pandas as pd


numerical_features = ['Elevation', 'Aspect', 'Slope']
categorical_features = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4']
target = 'Cover_Type'


def create_statistics( ):
    create_univarite_analysis()
    create_bivariate_analysis()


def create_univarite_analysis():
    create_numerical_analysis()
    create_categorical_analysis()


def create_numerical_analysis( data ):
    data[ numerical_features ].describe()
    data[ numerical_features ].hist()
    for num in numerical_features:
        plt.figure()
        plt.title( num )
        plt.boxplot( data[num], vert=False )


def create_categorical_analysis( data ):
    for cat in categorical_features:
        tmp = [cat, target]
        data.groupby(tmp)[target].count().unstack().plot(kind='bar', stacked=True)


def create_bivariate_analysis():
    create_feature_feature_analysis()
    create_feature_targets_analysis()
