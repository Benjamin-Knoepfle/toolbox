# -*- coding: utf-8 -*-
"""
Created on Wed May 20 22:16:00 2015

@author: root
"""
import sys
import logging


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import DBSCAN



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

def feature_importance( data ):
    features = data.drop( target, axis=1 )
    targets = data[target]
    if settings.Problem == 'Regression':
        clf = ensemble.ExtraTreesRegressor(n_estimators = 250, random_state=0, n_jobs=settings.NUMBER_OF_CORS)
    else:
        clf = ensemble.ExtraTreesClassifier(n_estimators = 250, random_state=0, n_jobs=settings.NUMBER_OF_CORS)
    clf.fit( features, targets )
    importances = clf.feature_importances_
    std = np.std( [tree.feature_importances_ for tree in clf.estimators_], axis=0 )
    indices = np.argsort(importances)[::-1]

    logger.info("Feature ranking")
    for f in range( len(features.columns) ):
        logger.info( "%d. feature %s (%f)" % (f+1, features.columns[indices[f]], importances[indices[f]]) )
    
    plt.figure()
    plt.title("Feature importance")
    plt.bar( range(len(features.columns)), importances[indices],
            color = 'r', yerr=std[indices], align='center')
    plt.xticks( range(len(features.columns)),  features.columns[indices], rotation=70)
    plt.xlim([-1, 10])
    plt.show()
    plt.savefig( settings.EXPO_FIGS+'Feature_importance.png' )
    #return zip(features.columns, clf.feature_importances_)


def cluster( data, eps=0.3, min_samples=10 ):
    db = DBSCAN( eps=eps, min_samples=min_samples ).fit(data)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    logger.info("Estimated number of clusters: %d" % n_clusters_ )
    
    unique_labels = set(labels)
    colors = plt.cm.Spectral( np.linspace(0,1, len(unique_labels)) )
    for k, col in zip(unique_labels, colors):
        if k==-1:
            col = 'k'
        class_member_mask = (labels == k)
        xy = data[class_member_mask & core_samples_mask]
        plt.plot( xy[:,0], xy[:,1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14 )
        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot( xy[:,0], xy[:,1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6 )
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    plt.savefig( settings.EXPO_FIGS+'Explorative_Clustering.png' )

def pca( data ):
    pca = PCA()
    features = numerical_features + categorical_features
    pca_data = pca.fit_transform( data[features] )
    pd.DataFrame( pca.explained_variance_ratio_ ).plot(kind='bar')
    plt.figure()
    plt.subplot(2,2,0)
    plt.scatter( pca_data[:,0], pca_data[:,1], c=data[target] )
    plt.subplot(2,2,1)
    plt.scatter( pca_data[:,2], pca_data[:,3], c=data[target] )
    plt.subplot(2,2,2)
    plt.scatter( pca_data[:,4], pca_data[:,5], c=data[target] )
    plt.subplot(2,2,3)
    plt.scatter( pca_data[:,6], pca_data[:,7], c=data[target] )
    return pca_data

def factor_analysis( data ):
    fa = FactorAnalysis()
    features = numerical_features + categorical_features
    fa_data = fa.fit_transform( data[features] )
    plt.figure()
    plt.subplot(2,2,0)
    plt.scatter( fa_data[:,0], fa_data[:,1], c=data[target] )
    plt.subplot(2,2,1)
    plt.scatter( fa_data[:,2], fa_data[:,3], c=data[target] )
    plt.subplot(2,2,2)
    plt.scatter( fa_data[:,4], fa_data[:,5], c=data[target] )
    plt.subplot(2,2,3)
    plt.scatter( fa_data[:,6], fa_data[:,7], c=data[target] )
    return fa_data
    

def create_histogram( data, feature, bins=10 ):
    plt.figure()
    plt.title( feature )
    plt.hist( data[feature], bins=bins )
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
    data = pd.read_csv( settings.PROCESSED_TRAIN, delimiter=',' )
    create_univarite_analysis()
    create_bivariate_analysis()
