#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:02:30 2020

@author: ben

Create a new version of the prototype random forest without 'Open' input
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
del input_data['Open']
print(input_data)

# Set output variables
labels = np.array(data['4. close'])
attribute_list = list(input_data.columns)
labels = np.delete(labels, 0)



# Use scitkit learn to split the data
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(input_data, labels, test_size = 0.2)

# Train model
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



# Variable importance
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(attribute_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

#print variable importance 
print('\n')
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# VISUALISE
import matplotlib.pyplot as plt

# =============================================================================
# # Plot pie chart
# 
# pie_labels = attribute_list
# pie_values = importances
# 
# df = pd.DataFrame(
#     data = {'attribute': pie_labels, 'value':pie_values})
# print(df)
# for index, row in df.iterrows():
#     if row['value'] < 0.01:
#         df.drop(row[])
# 
# fig, ax = plt.subplots()
# ax.pie(pie_values, labels = pie_labels, startangle=90, textprops={'fontsize': 10})
# ax.axis('equal')
# #ax.legend(loc=1, labels=pie_labels, prop={'size': 12})
# 
# plt.tight_layout()
# plt.show()
# =============================================================================


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
    



