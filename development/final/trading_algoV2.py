#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:35:15 2021

@author: ben

Trading algo - Version 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import os
from sklearn.model_selection import RandomizedSearchCV, validation_curve, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix    
from sklearn.ensemble import RandomForestClassifier
import math


'''
Replace the line below with the file path of this project's location.
Use '/' as the folder seperator
'''
YOUR_PATH = 'YOUR_PATH'


'''
Call best random forest model
'''
filepath = YOUR_PATH+'/models/ml_model'
rf_grid = pickle.load(open(filepath, 'rb'))
rf_model = rf_grid.best_estimator_


'''
Process data from existing CSV created in 'tech_inds_classifier'
'''
data = pd.read_csv(YOUR_PATH+'/input-data/full_data.csv')
data['Date'] = data['Unnamed: 0']
data = data.drop(['Unnamed: 0'], axis=1)
data = data.set_index(['Date'])
data.index = pd.to_datetime(data.index)           # Setting date as entry index
data = data.dropna().copy()
test_data = data.loc['2019-01-01':'2019-12-31']   # Store data for dates we will be running the algo on


'''
Begin trading algorithm v2
'''

'''
Function:
    get the top n predictions returned by the ml model
'''
def get_top_predictions(date, n):  
    day_data = data.loc[date]
    predictions = pd.DataFrame({'ticker':day_data.ticker, 
                                'up prob':rf_model.predict_proba(day_data.drop(['ticker'],axis=1))[:,1]})
    top_predictions = predictions.sort_values(by=['up prob'], ascending=False)
    top_predictions = top_predictions.loc[top_predictions['up prob']>= 0.6]     # only allow if confidence > 60%
    if (n<10):
        return top_predictions[:n]
    else:
        return top_predictions[:10] # max 10 trades at a given time
 
    '''
Function:
    Create an instance of a trade in the algo dataframe
'''
def open_trade(day, index, trade, split):
    open_price = float(data.loc[(data.index==day) & (data['ticker']==trade['Ticker'])]['Open'])
    quantity = math.floor(split/open_price)
    trade_price = open_price*quantity
    # update df values for an open trade
    algo_df.loc[algo_df.index == index, 'Open_date'] = day
    algo_df.loc[algo_df.index == index, 'Open_price'] = open_price
    algo_df.loc[algo_df.index == index, 'Quantity'] = quantity
    algo_df.loc[algo_df.index == index, 'Trade_price'] = trade_price
    algo_df.loc[algo_df.index == index, 'Expiry'] = 9
    algo_df.loc[algo_df.index == index, 'Status'] = 'open'
    
    # update account numbers
    global cash
    cash -= trade_price

'''
Function:
    return true if stop loss is reached, false otherwise
'''
def stop_loss(day, trade):
    limit = trade.Open_price * 0.95
    # return true if either open or low price falls below stop loss limit
    return ((data.loc[(data.index == day) & (data['ticker'] == trade['Ticker']), 'Open']<limit) |
            (data.loc[(data.index == day) & (data['ticker'] == trade['Ticker']), 'Low']<limit))

'''
Function:
    Update trade info in dataframe to apply 'closing' results
'''
def close_trade(day, trade):
    close_price = float(data.loc[(data.index==day) & (data['ticker']==trade['Ticker'])]['Close']) 
    profit = close_price*trade['Quantity']-trade['Trade_price']
    # update df values for a closed trade
    algo_df.loc[algo_df.index == index, 'Close_date'] = day
    algo_df.loc[algo_df.index == index, 'Close_price'] = close_price
    algo_df.loc[algo_df.index == index, 'Profit'] = profit
    algo_df.loc[algo_df.index == index, 'Expiry'] = 0
    algo_df.loc[algo_df.index == index, 'Status'] = 'closed'
    
    #update account info 
    global equity, cash
    equity += profit      
    cash += close_price*trade['Quantity']
    
    # update k info 
    global k_info
    if (profit>0):
        k_info[0] += 1          # Increase first 2 entries if profitable
        k_info[1] += profit
    else:
        k_info[2] += 1          # Increase last 2 entries if loss
        k_info[3] += abs(profit)
    
    '''
Function:
    Update the k% by performing the approriate calculations of the k array
'''
def update_k(k, k_info):
    W = k_info[0]/(k_info[0]+k_info[2])     # % of won trades
    R = k_info[1]/k_info[3]                 # Ratio of win amount:loss amount
    k = W - (1-W)/R
    if (k>0.1):     # cap k at 0.1
        k=0.1
    return k


i=0
dates = test_data.index.unique()    # list of dates
equity = 1000000                    # equity: total account amount/worth
cash = equity                       # cash: equity available to spend 
algo_df = pd.DataFrame(columns =['Ticker',
                                       'Open_date',
                                       'Open_price',
                                       'Quantity',
                                       'Trade_price',
                                       'Close_date',
                                       'Close_price',
                                       'Profit',
                                       'Expiry',
                                       'Status'])            # dataframe to store info and track each transaction

k = 0.05                # start k = 5%
k_info = [0,0,0,0]      # [wins, win amounts, losses, loss amounts]
for i in range(len(dates)):     # for every day in the test year
    day = dates[i]              # update day
    split = equity*k            # split: amount of equity to spend on one trade
    
    # Buy pending trades
    for index, row in algo_df.loc[algo_df['Status']=='pending'].iterrows():
        open_trade(day, index, row, split)
    
    # Update (and possibly close) open trades
    for index, row in algo_df.loc[algo_df['Status']=='open'].iterrows():
        if ((stop_loss(day, row).bool()) | (row.Expiry <= 0)):      # Check if stop-loss is reached or trade has expired
            close_trade(day, row)
        else:
            algo_df.loc[algo_df.index == index, 'Expiry'] -= 1      # decrease expiry by one (since one day has now passed)
    
    # Update k
    if (len(algo_df.index)>=40):    # keep k=0.05 until enough info has gathered
        k = update_k(k, k_info)
    split = equity*k
    
    # Check for new trades to execute if criteria is met
    if ((len(algo_df.loc[algo_df['Status']=='open'].index)<10) & (cash>split)):  # cap at 10 open trades allowed, must have enough cash to buy
        num_of_new_trades = math.floor(cash/split)
        top_predictions = get_top_predictions(day, num_of_new_trades)       # get top trades for appropriate number
        for index, pred in top_predictions.iterrows():
            trade_df = pd.DataFrame({'Ticker': [pred['ticker']],
                                       'Open_date': [''],
                                       'Open_price': [''],
                                       'Quantity': [''],
                                       'Trade_price': [''],
                                       'Close_date': [''],
                                       'Close_price': [''],
                                       'Profit': [''],
                                       'Expiry': [''],
                                       'Status': ['pending']})
            algo_df = algo_df.append(trade_df, ignore_index=True)       
    
print("Equity: ",equity)
print("K_info list [wins, win amount, losses, loss amount]: ",k_info)
algo_df.to_csv(YOUR_PATH+'/algo_dfV2-2019.csv', encoding='utf-8')

