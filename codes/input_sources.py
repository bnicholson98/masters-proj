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


'''Basic price Data'''
# =============================================================================
# ts = TimeSeries(key=MY_KEY, output_format='pandas')
# data, meta_data = ts.get_daily(symbol='GOOGL',outputsize='full')
# data.to_csv('/home/ben/Documents/masters-proj/input-data/GOOG_prices.csv')
# =============================================================================


'''Tech inds'''
ti = TechIndicators(key = MY_KEY, output_format='pandas')

# SMA
# =============================================================================
# ti_data, ti_meta = ti.get_sma(symbol='GOOGL')
# ti_data.to_csv('/home/ben/Documents/masters-proj/input-data/GOOG_sma.csv')
# =============================================================================

# EMA
# =============================================================================
# ti_data, ti_meta = ti.get_ema(symbol='GOOGL')
# ti_data.to_csv('/home/ben/Documents/masters-proj/input-data/GOOG_ema.csv')
# =============================================================================

# MACD
# =============================================================================
# ti_data, ti_meta = ti.get_macd(symbol='GOOGL')
# ti_data.to_csv('/home/ben/Documents/masters-proj/input-data/GOOG_macd.csv')
# =============================================================================

# Stoch
# =============================================================================
# ti_data, ti_meta = ti.get_stoch(symbol='GOOGL')
# ti_data.to_csv('/home/ben/Documents/masters-proj/input-data/GOOG_stoch.csv')
# =============================================================================

# RSI
# =============================================================================
# ti_data, ti_meta = ti.get_rsi(symbol='GOOGL')
# ti_data.to_csv('/home/ben/Documents/masters-proj/input-data/GOOG_rsi.csv')
# =============================================================================

# ADX
# =============================================================================
# ti_data, ti_meta = ti.get_adx(symbol='GOOGL')
# ti_data.to_csv('/home/ben/Documents/masters-proj/input-data/GOOG_adx.csv')
# =============================================================================

# AROON
# =============================================================================
# ti_data, ti_meta = ti.get_aroon(symbol='GOOGL')
# ti_data.to_csv('/home/ben/Documents/masters-proj/input-data/GOOG_aroon.csv')
# =============================================================================

# BBands
# =============================================================================
# ti_data, ti_meta = ti.get_bbands(symbol='GOOGL')
# ti_data.to_csv('/home/ben/Documents/masters-proj/input-data/GOOG_bbands.csv')
# =============================================================================

# AD
# =============================================================================
# ti_data, ti_meta = ti.get_ad(symbol='GOOGL')
# ti_data.to_csv('/home/ben/Documents/masters-proj/input-data/GOOG_ad.csv')
# =============================================================================

# OBV
# =============================================================================
# ti_data, ti_meta = ti.get_obv(symbol='GOOGL')
# ti_data.to_csv('/home/ben/Documents/masters-proj/input-data/GOOG_obv.csv')
# =============================================================================




# =============================================================================
# data['4. close'].plot()
# =============================================================================
# # plt.title('Intraday TimeSeries Google')
# =============================================================================
# plt.ylabel("Price ($)")
# plt.xlabel("Year")
# plt.show()
# =============================================================================


