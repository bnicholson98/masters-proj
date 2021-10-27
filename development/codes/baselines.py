#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:06:56 2021

@author: ben

Base models
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
print(input_data)

# Set output variables
labels = np.array(data['4. close'])
attribute_list = list(input_data.columns)
labels = np.delete(labels, 0)



'''
Constant change baseline
'''
# Use scitkit learn to split the data
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(input_data, labels, test_size = 0.2)

# perform baseline predictions
const_base_preds =np.array([])

for index, row in test_data.iterrows():
    pred = row[3]-row[0]+row[8]
    const_base_preds = np.append(const_base_preds, pred)

# eval baseline model
print("Evaluation of constant change baseline model")
errors = abs(const_base_preds - test_labels)
print("Avg error: ", round(np.mean(errors),2))

# direction prediction accuracy
correct__dir_guess = 0

for i in range(len(test_labels)):
    if (test_labels[i] >= test_data['Open'][i] and const_base_preds[i] >= test_data['Open'][i]) or \
       (test_labels[i] < test_data['Open'][i] and const_base_preds[i] < test_data['Open'][i]):
           correct__dir_guess += 1

dir_guess_accuracy = (correct__dir_guess/len(test_labels))*100
print('Accuracy of direction prediction: ', round(dir_guess_accuracy, 2),'%')          
    

'''
No price move baseline
'''
no_move_base_preds = np.array([])

for index, row in test_data.iterrows():
    pred = row[8]
    no_move_base_preds= np.append(no_move_base_preds, pred)
    
# eval no move baseline
print("\n")
errors = abs(no_move_base_preds - test_labels)
print("Avg error on 'no move' baseline: ",round(np.mean(errors),2))


'''
Always up baseline
'''
correct__dir_guess = 0

for i in range(len(test_labels)):
    if (test_labels[i] >= test_data['Open'][i]):
        correct__dir_guess += 1

dir_guess_accuracy = (correct__dir_guess/len(test_labels))*100
print('\n')
print("Accuracy of 'always up' prediction: ", round(dir_guess_accuracy, 2),'%') 
