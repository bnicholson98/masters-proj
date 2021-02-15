#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:39:01 2020

@author: ben
"""


# =============================================================================
# # Yahoo data input
# 
# import yfinance as yf
# 
# data = yf.download('AAPL', '2019-01-01', interval='1h')
# 
# data.Close.plot()
# plt.show()
# =============================================================================


# =============================================================================
# # Quandl data input 
# import quandl
# quandl.ApiConfig.api_key = 'ZLWsMWNaYaQuhKMyyyMY'
# data = quandl.get('CHRIS/MGEX_IH1')
# data.plot() 
# plt.show() 
# =============================================================================


MY_KEY = '54SG4FMYHLXQJZ56'
# Alpha Vanatge input
import matplotlib.pyplot as plt 
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators


ts = TimeSeries(key=MY_KEY, output_format='pandas')
data, meta_data = ts.get_daily(symbol='GOOGL',outputsize='full')
print(data)

ti = TechIndicators(key = MY_KEY, output_format='pandas')
ti_data, ti_meta = ti.get_sma(symbol='GOOGL')
print(ti_data)
print(ti_meta)
data['4. close'].plot()
plt.title('Intraday TimeSeries Google')
plt.ylabel("Price ($)")
plt.xlabel("Year")
plt.show()


