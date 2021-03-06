#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:43:32 2020

@author: ben

Initial prototype - Gradient boosting model
"""
import matplotlib.pyplot as plt 
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np

# Call Google price info from AlphaVantage
ts = TimeSeries(key='54SG4FMYHLXQJZ56',output_format='pandas')
data, meta_data = ts.get_daily(symbol='GOOGL',outputsize='full')

# Data preparation
data = data['2019-12-31':'2016-01-01']                      
data = data.iloc[::-1]
data['Day'] = data.index.day
data['Month'] = data.index.month
data['Year'] = data.index.year

input_data = data.copy()
input_data = input_data.shift(periods = 1)
input_data['Open'] = data['1. open']
input_data.columns= ['Prev. Open','Prev. High', 'Prev. Low', 'Prev. Close', 'Prev. Volume', 'Day', 'Month', 'Year', 'Open']
#print(input_data)
input_data = input_data.drop(input_data.index[0])
#print(input_data)

# Set output variables
labels = np.array(data['4. close'])
attribute_list = list(input_data.columns)
labels = np.delete(labels, 0)



# Use scitkit learn to split the data
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(input_data, labels, test_size = 0.2)

# Train model
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01)
gb.fit(train_data, train_labels)

# test model
predictions = gb.predict(test_data)



# Error
errors = abs(predictions - test_labels)
print("Avg error on model: ", round(np.mean(errors),2))

# Mape
mape = 100*(errors/test_labels)
accuracy = 100 - np.mean(mape)
print("Accuracy: ", round(accuracy, 2),"%")

'''
Test to predict direction accuracy
'''
correct__dir_guess = 0

for i in range(len(test_labels)):
    if (test_labels[i] >= test_data['Open'][i] and predictions[i] >= test_data['Open'][i]) or \
       (test_labels[i] < test_data['Open'][i] and predictions[i] < test_data['Open'][i]):
           correct__dir_guess += 1

dir_guess_accuracy = (correct__dir_guess/len(test_labels))*100
print('Accuracy of direction prediction: ', round(dir_guess_accuracy, 2),'%')          
    
# Example prediction  
print('\n')    
print(test_data['Open'][100])
print(test_labels[100])
print(predictions[100])


print()
# Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import mean
from numpy import std

cv = RepeatedKFold(n_splits=10, n_repeats=5)
# evaluate the model
n_scores = cross_val_score(
    gb, input_data, labels,
    scoring='neg_mean_absolute_error',
    cv=cv, n_jobs=-1)
# report performance
print("10-fold evaluation of gradient boosting model:")
print('Mean Average Error: %.2f' % (mean(n_scores)*(-1)))
print('Standard deviation: %.2f' % std(n_scores))

