#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:42:56 2021

@author: ben

script to run and save the random forest model on the given data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import os
from sklearn.model_selection import RandomizedSearchCV, validation_curve, TimeSeriesSplit, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

'''
Replace the line below with the file path of this project's location.
Use '/' as the folder seperator
'''
YOUR_PATH = 'YOUR_PATH'


'''
Process data from existing CSV created in 'tech_inds_classifier'
'''
data = pd.read_csv(YOUR_PATH+'/input-data/full_data.csv')
data['Date'] = data['Unnamed: 0']
data = data.drop(['Unnamed: 0'], axis=1)
data = data.set_index(['Date'])
data.index = pd.to_datetime(data.index)

'''
Output label
'''
data['shifted'] = data.groupby('ticker')['Close'].transform(lambda x: x.shift(-9))
data['direction'] = np.where(data['Open']<data['shifted'], 1,0)
data = data.drop(['shifted'], axis=1)
data = data.loc[:'2019-12-31']
data = data.dropna().copy()

'''
Create ML model
'''
labels = data['direction']
data = data.drop(['direction'], axis=1)
data = data.drop(['ticker'], axis=1)




'''
Below is commented out code used for the creation of validation curves
'''
# =============================================================================
# '''
# Validation curve
# '''
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import validation_curve
# from sklearn.model_selection import TimeSeriesSplit
# 
# #Create validation curve for the Random Forest Classifier
# rf = RandomForestClassifier()
# train_scoreNum, test_scoreNum = validation_curve(rf,
#                                 X = data, y = labels, 
#                                 param_name = 'n_estimators', 
#                                 param_range = [50,100,200,500], 
#                                 cv = TimeSeriesSplit(n_splits = 3),
#                                 verbose=3)
# 
# train_scores_mean = np.mean(train_scoreNum, axis=1)
# train_scores_std = np.std(train_scoreNum, axis=1)
# test_scores_mean = np.mean(test_scoreNum, axis=1)
# test_scores_std = np.std(test_scoreNum, axis=1)
# 
# plt.figure(figsize = (20,10))
# plt.plot([50,100,200,500],train_scores_mean)
# plt.plot([50,100,200,500],test_scores_mean)
# plt.legend(['Train Score','Test Score'], fontsize = 'large')
# plt.xlabel('n_estimators')
# plt.ylabel('Score')
# plt.title('Validation Curve Score for n_estimators', fontsize = 'large')
# 
# 
# 
# print(train_scores_mean)
# print(train_scores_std)
# print(test_scores_mean)
# print(test_scores_std)
# =============================================================================


    
'''
Below is commented code for the grid search, this will need re-running for a new model to be made
'''
'''
Grid Search
'''
# =============================================================================
# '''
# Train model
# '''
# # Split the data into test and train
# train_data = data.loc[:'2018-12-31']
# test_data = data.loc['2019-01-01':]
# train_labels = labels.loc[:'2018-12-31']
# test_labels = labels.loc['2019-01-01':]
# 
# # Hyperparameters
# params = {'n_estimators': [80,100,120],
#           'max_features':['sqrt','log2'],
#           'max_depth': [40,50,60,None],
#           'criterion': ['gini','entropy']}
# 
# rf = RandomForestClassifier()
# time_series_split = TimeSeriesSplit(n_splits = 3)
# 
# # Create gridsearch for hyperparam tuning
# rf_cv = GridSearchCV(rf, params, cv = time_series_split, n_jobs = -1, verbose = 20)
# 
# #Fit the random forest with our X_train and Y_train
# rf_cv.fit(train_data, train_labels)
#       
# #Save the fitted variable into a Pickle file
#     
# pickle.dump(rf_cv, open('ml_model','wb'))
# =============================================================================



# Train model
# Split the data into test and train
# Use scitkit learn to split the data
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2)

train_data = data.loc[:'2018-12-31']
test_data = data.loc['2019-01-01':]
train_labels = labels.loc[:'2018-12-31']
test_labels = labels.loc['2019-01-01':]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100)

rf.fit(train_data, train_labels)

'''
Predicting
'''
predictions = rf.predict(test_data)



'''
Evaluation
'''
from sklearn.metrics import accuracy_score         
    
accuracy= 100*accuracy_score(test_labels, predictions)
print("Prediction accuracy: ",accuracy,'%')

### confusion matrix  ####
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(test_labels, predictions)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(rf, test_data, test_labels)
plt.show()


# Variable importance
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(attribute_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# VISUALISE
import matplotlib.pyplot as plt

#set style
plt.style.use('fivethirtyeight')
# list x locations
x_values = list(range(len(importances)))


#make bar chart
plt.bar(x_values, importances, orientation='vertical')
# Tick labels
plt.xticks(x_values, attribute_list, rotation='vertical')
# Axis and title labels
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()
    
