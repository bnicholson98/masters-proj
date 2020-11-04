#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:33:11 2020

@author: ben
"""


from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override()

import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)

msft = yf.Ticker("MSFT")

data = pdr.get_data_yahoo(tickers = "MSFT", period="max", group_by = "ticker")
data['Day'] = data.index.day
data['Month'] = data.index.month
data['Year'] = data.index.year

# Set labels (output var)
labels = np.array(data["Close"])
# Remove labels from data
data = data.drop("Close", axis=1)
data = data.drop("Adj Close", axis=1)
attribute_list = list(data.columns)
print(attribute_list)
data = np.array(data)

# Use skikit learn to split data
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2)

# Set baseline prediction (avg between high and low)
baseline_preds = np.array([])
for i in test_data:
    avg = (i[1]+i[2])/2
    baseline_preds = np.append(baseline_preds, avg)

baseline_errors = abs(baseline_preds-test_labels)
print('Avg baseline error: ', round(np.mean(baseline_errors),2))


# train model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 500)
rf.fit(train_data, train_labels)

# test model
predictions = rf.predict(test_data)

# error
errors = abs(predictions - test_labels)
print("Avg error on model: ", round(np.mean(errors),2))

# mape
mape = 100*(errors/test_labels)
accuracy = 100 - np.mean(mape)
print("Accuracy: ", round(accuracy, 2),"%")

# interpret
from sklearn.tree import export_graphviz
import pydot

# example of shortened tree
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 5)
rf_small.fit(train_data, train_labels)
tree_small = rf_small.estimators_[5]
# save tree example
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = attribute_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');

# variable importance
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(attribute_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# visualise
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


# ATTEMPT OF GRAPH
# Use datetime for creating date objects for plotting
import datetime
# Dates of training values
months = data[:, attribute_list.index('Month')]
days = data[:, attribute_list.index('Day')]
years = data[:, attribute_list.index('Year')]
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# Dates of predictions
months = test_data[:, attribute_list.index('Month')]
days = test_data[:, attribute_list.index('Day')]
years = test_data[:, attribute_list.index('Year')]
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
plt.xlabel('Date'); plt.ylabel('Close Price($)'); plt.title('Actual and Predicted Values');




