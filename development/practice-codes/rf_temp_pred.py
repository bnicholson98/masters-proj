#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 12:41:57 2020

@author: ben
"""


import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

features = pd.read_csv('temps.csv')

print(features.shape)
features.describe()

# One hot encoding
features = pd.get_dummies(features)
features.iloc[:,5:].head(5)
# set labels
labels = np.array(features['actual'])
# remove labels from features
features = features.drop('actual', axis=1)
feature_list = list(features.columns)
features = np.array(features)

# Use ski-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data 
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

# baseline predicitons
baseline_preds = test_features[:, feature_list.index('average')]
baseline_errors = abs(baseline_preds-test_labels)
print('Avg baseline error: ', round(np.mean(baseline_errors),2))


# TRAIN MODEL
# import model
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees    
rf = RandomForestRegressor(n_estimators = 1000,random_state = 42)
# train model on training data
rf.fit(train_features, train_labels)

# TEST MODEL
# predict on test data
predictions = rf.predict(test_features)
# calc errors
errors = abs(predictions - test_labels)
# print avg
print("Mean abs error on model: ", round(np.mean(errors),2))


# mean absolute perentage error
mape = 100*(errors/test_labels)
# calc and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy: ',round(accuracy, 2),'%')


### Improve model here if necessary ###

# INTERPRET MODEL
# look at one decision tree
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot


# see example of 'shortened' tree
# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');

# variable importance
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
## [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


#### Reattempt model without useless variables

# VISUALISATIONS
import matplotlib.pyplot as plt
# for notebook %matplotlib inline

#set style
plt.style.use('fivethirtyeight')
# list x locations
x_values = list(range(len(importances)))
#make bar chart
plt.bar(x_values, importances, orientation='vertical')
# Tick labels
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis and title labels
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
plt.show()

# Use datetime for creating date objects for plotting
import datetime
# Dates of training values
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# Dates of predictions
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]
# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()

# Graph labels
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');



