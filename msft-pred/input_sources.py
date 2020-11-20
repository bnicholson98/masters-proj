#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:39:01 2020

@author: ben
"""


# Yahoo data input

import matplotlib.pyplot as plt 
import yfinance as yf

data = yf.download('AAPL', '2019-01-01', interval='1h')

data.Close.plot()
plt.show()


# Quandl data input 
import quandl
quandl.ApiConfig.api_key = 'ZLWsMWNaYaQuhKMyyyMY'
data = quandl.get('CHRIS/MGEX_IH1')
data.plot() 
plt.show() 


# Alpha Vanatge input
from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key='54SG4FMYHLXQJZ56',output_format='pandas')
data, meta_data = ts.get_intraday(symbol='GOOGL',interval='1min', outputsize='full')
print(data)

data['4. close'].plot()
plt.title('Intraday TimeSeries Google')
plt.show()