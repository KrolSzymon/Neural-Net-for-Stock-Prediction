import yfinance as yf
import pandas as pd
import numpy as np

snp = yf.Ticker("SWPPX")
stock = yf.Ticker("BTC-USD")
stock_history = stock.history('max')
snp_history = snp.history('max')

snp_history.index = snp_history.index.tz_localize(None)
stock_history.index = stock_history.index.tz_localize(None)
first_stock_day = stock_history.index[0] 
if(first_stock_day>snp_history.index[0]):
    stock_start = stock_history.head(1).index
    stock_start = stock_start.strftime("%Y-%m-%d")
    stock_start = stock_start[0]
    snp_history = snp_history[stock_start:]



