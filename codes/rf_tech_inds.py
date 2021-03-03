
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:58:09 2021

@author: ben

Random forest with prices and tech indicators
"""


import matplotlib.pyplot as plt 
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import glob
import os


'''
Data preparation
'''
path = '/home/ben/Documents/masters-proj/input-data/'                   # path containing input data           
all_files = glob.glob(os.path.join(path, "*.csv"))                      # gather all csv files
data = pd.merge(pd.read_csv(all_files[0]),pd.read_csv(all_files[1]))    # start csv merge

for i  in range(len(all_files)-2):                                      # loop through and merge all csv files into one dataframe
    data = pd.merge(data, pd.read_csv(all_files[i+2]))    


data['date'] = pd.to_datetime(data['date'], utc=False)      # set date column as pandas date object
data.set_index('date', inplace=True)                        # set date as index
data = data['2010-01-01':'2019-12-31']                      # subset only data from 2010-2019                                                        
data['Day'] = data.index.day                                # create day column
data['Month'] = data.index.month                            # ... month column
data['Year'] = data.index.year                              # ... year column
# Reorder columns
data = data[['Day', 'Month', 'Year','1. open', '2. high', '3. low', '4. close', '5. volume', 'SlowD', 'SlowK', 'Chaikin A/D', 'ADX', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA', 'Aroon Up', 'Aroon Down', 'EMA', 'OBV', 'Real Lower Band', 'Real Upper Band', 'Real Middle Band', 'RSI']]

input_data = data.copy()                            # copy to new dataframe
input_data = input_data.shift(periods = 1)          # shift results down 1
input_data['Open'] = data['1. open']                # set today open for previous day date
# rename the columns
input_data.columns= ['Day', 'Month', 'Year','Prev Open', 'Prev High', 'Prev Low', 'Prev Close', 'Prev Volume', 'SlowD', 'SlowK', 'Chaikin A/D', 'ADX', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SMA', 'Aroon Up', 'Aroon Down', 'EMA', 'OBV', 'Real Lower Band', 'Real Upper Band', 'Real Middle Band', 'RSI', 'Open'] # give columns appropriate names
input_data = input_data.drop(input_data.index[0])   # remove the top result (now containing NaN inputs)

print(input_data)
    
# Set output variables
_open_ = np.array(data['1. open'])          # take day open prices
_close_ = np.array(data['4. close'])        # take day close prices
labels = np.array([])
attribute_list = list(input_data.columns)   # list columns as attribute names
for i in range(len(_open_)):                # for each day, set +1 if day was an up day, and -1 otherwise
    if (_open_[i] <= _close_[i]):
        labels = np.append(labels, 1)
    else:
        labels = np.append(labels, 0)
labels = np.delete(labels, 0)               # remove top result (due to previous shift)


'''
Learning
'''
# Use scitkit learn to split the data
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(input_data, labels, test_size = 0.2)

# Train model
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
# Example prediction  
print('\n')    
print(test_data['Open'][100])
print(test_labels[100])
print(predictions[100])



'''
Variable importances
'''
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(attribute_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

#print variable importance 
print('\n')
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

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
    


### confusion matrix  ####
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(test_labels, predictions)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(rf, test_data, test_labels)
plt.show()

















