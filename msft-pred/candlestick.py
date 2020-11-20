#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:44:48 2020

@author: ben
"""


# Alpha Vanatge input
import matplotlib.pyplot as plt 
from alpha_vantage.timeseries import TimeSeries
import pandas as pd

ts = TimeSeries(key='54SG4FMYHLXQJZ56',output_format='pandas')
data, meta_data = ts.get_daily(symbol='GOOGL',outputsize='full')
data = data['2019-12-31':'2016-01-01']
print(data.head())

data['4. close'].plot()
plt.title('Day TimeSeries Google')
plt.show()


### SECTORS DATA ###
from alpha_vantage.sectorperformance import SectorPerformances

sp = SectorPerformances(key='54SG4FMYHLXQJZ56',output_format='pandas')
sector, meta_data = sp.get_sector()
print(sector)
sector['Rank I: Year Performance'].plot(kind='bar')
plt.title('5 Year Performance (%) per Sector')
plt.tight_layout()
plt.grid()
plt.show()


### CANDLESTICK GRAPH ###

dates = (pd.to_datetime(data.index.values))
data['6. date'] = dates


fig, ax = plt.subplots(figsize=(16,8))
plt.plot(data['6. date'], data['4. close'], c='black')
plt.grid()
plt.show()

# Pre-process data for interpretation
del data['6. date']
data.columns= ['Open','High', 'Low', 'Close', 'Volume']
data = data['2019-12-31':'2019-12-01']
data = data.iloc[::-1]

# MPLFINANCE
import mplfinance as fplt

plt = fplt.plot(data,
           type='candle',
           style = 'yahoo',
           title = 'GOOGL - Dec 2019',
           ylabel='Price ($)',
           #volume = True,
           #ylabel_lower = 'Volume'
           )











