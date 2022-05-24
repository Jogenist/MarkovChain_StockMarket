# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
#Data Source
import yfinance as yf
#Data viz
import plotly.graph_objs as go

#Interval required 5 minutes
df = yf.download(tickers='SIE.DE', period='1y', interval='1wk')
#Print data
print(df)

#calculate percentages and drop first row
df = df.pct_change()
df = df.iloc[1: , :]
#Print data
print(df)

#binning data (high medium low)



