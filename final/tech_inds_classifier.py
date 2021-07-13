#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:52:36 2021

@author: ben

This script prepares a CSV file for the models input data by creating the tech inds
"""
# imports
import requests
import bs4 as bs
import numpy as np
import pandas_datareader as pdr
import pandas as pd
import pandas_ta as ta

'''
Replace the line below with the file path of this project's location.
Use '/' as the folder seperator
'''
YOUR_PATH = 'YOUR_PATH'



# collecting S&P 100 tickers
wiki = requests.get('https://en.wikipedia.org/wiki/S%26P_100')
content = wiki.content
soup = bs.BeautifulSoup(content, 'lxml')
soup = soup.find('table',{'class':'wikitable sortable'})
tickers = np.array([])

for rows in soup.findAll('tr')[1:]:
    ticker = rows.findAll('td')[0].text.strip()
    tickers = np.append(tickers, ticker)

data = pd.DataFrame()
extra_data=pd.DataFrame()
missing_data = []
pd.set_option('mode.chained_assignment', None)

#get price data
for ticker in tickers:
    try:
        price_data = pdr.get_data_yahoo(ticker, start="2009-11-01", end='2020-01-30')
        price_data['ticker'] = ticker
        data = data.append(price_data)
    except:
        missing_data.append(ticker)

# create technical indicators
data['10 day'] = data.groupby('ticker')['Close'].pct_change(periods=10) # 10 day price change
data['5 day'] = data.groupby('ticker')['Close'].pct_change(periods=5)   # 5 day price change 

data['sma15'] = data.groupby('ticker')['Close'].transform(lambda x: x.rolling(window=10).mean()) # Simple moving average

data['ema15'] = data.groupby('ticker')['Close'].transform(lambda x: x.ewm(span=15, adjust=False).mean()) # Exponential moving average


data['middleband'] = data.groupby('ticker')['Close'].transform(lambda x: x.rolling(window=20).mean())   # middle bollinger band
extra_data['SD'] = data.groupby('ticker')['Close'].transform(lambda x: x.rolling(window=20).std())
data['upperband'] = data['middleband'] + 2*extra_data['SD']                                             # upper bollinger band
data['lowerband'] = data['middleband'] - 2*extra_data['SD']                                             # lower bollinger band


temp_data = data.copy()
aroon_data = pd.DataFrame()
for i in data['ticker'].unique():
    aroon_data = aroon_data.append(data.loc[data.ticker==i].ta.aroon())      # AROON indicators
data = pd.concat([data, aroon_data], axis=1)


data['ema12'] = data.groupby('ticker')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())    # 12 day ema
data['ema26'] = data.groupby('ticker')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())    # 26 day ema
data['MACD'] = data['ema12'] - data['ema26']                                                                # MACD indicator


# OBV indicator
for i in data['ticker'].unique():
    data.loc[data.ticker==i, 'OBV'] = np.where(data.loc[data.ticker==i, 'Close'] > data.loc[data.ticker==i, 'Close'].shift(1),
                                               data.loc[data.ticker==i, 'Volume'],
                                                np.where(data.loc[data.ticker==i,'Close'] < data.loc[data.ticker==i, 'Close'].shift(1),
                                                         -data.loc[data.ticker==i, 'Volume'], 0)).cumsum()

# Stochastic indicator
extra_data['lowest14'] = data.groupby('ticker')['Low'].transform(lambda x: x.rolling(window=14).min())
extra_data['highest14'] = data.groupby('ticker')['High'].transform(lambda x: x.rolling(window=14).max())
data['StochK'] = 100*((data['Close']-extra_data['lowest14'])/(extra_data['highest14']-extra_data['lowest14']))
data['StochD'] = data.groupby('ticker')['StochK'].transform(lambda x: x.rolling(window=3).mean())


# rsi indicator
for i in data['ticker'].unique():
    data.loc[data.ticker==i, 'RSI'] = data.loc[data.ticker==i].ta.rsi()

adx_data = pd.DataFrame()
# ADX indicator
for i in data['ticker'].unique():
    adx_data = adx_data.append(data.loc[data.ticker==i].ta.adx())
data['ADX_14'] = adx_data['ADX_14']

# CMF 
for i in data['ticker'].unique():
    data.loc[data.ticker==i, 'CMF'] = data.loc[data.ticker==i].ta.cmf()
# Chaikin Oscillator
temp_data['AD'] = (2*data['Close']-data['High']-data['Low'])/(data['High']-data['Low']) *data['Volume']
data['Chaikin'] = pd.Series(temp_data.groupby('ticker')['AD'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
                            - temp_data.groupby('ticker')['AD'].transform(lambda x: x.ewm(span=10).mean()))

data = data.loc['2010-01-01':]  # Take time frame 2010-2019
print(data)
data.to_csv(YOUR_PATH+'/input-data/full_data.csv', encoding='utf-8')