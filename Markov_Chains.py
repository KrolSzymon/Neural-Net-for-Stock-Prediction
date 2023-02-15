import matplotlib
import io, base64, os, json, re
import pandas as pd
import numpy as np
import datetime
from random import randint

dataset_address = 'MCD.csv'
cut_off_date = '2010-01-01'
number_of_days = 100
data = pd.read_csv(dataset_address)
print(data.head())
data['Date'] = pd.to_datetime(data['Date'])
rows_per_sequence = 10
data_in_date = data[data['Date'] >= cut_off_date]

print(data_in_date.head())

new_row_set = []
start_row = 0
counter = 0
for row_set in range(0,number_of_days):
    
    start_row = counter * rows_per_sequence
    
    end_row = start_row + rows_per_sequence
    market_subset = data_in_date.iloc[start_row : end_row]
    close_date = max(market_subset['Date'])
    
    close_gap = market_subset['Close'].pct_change()
    high_gap = market_subset['High'].pct_change()
    low_gap = market_subset['Low'].pct_change()
    volume_gap = market_subset['Volume'].pct_change()
    daily_change = (market_subset['Close'] - market_subset['Open']) / market_subset['Open']
    close_price_next_day_direction = (market_subset['Close'].shift(-1) - market_subset['Close'])
    
    new_row_set.append(pd.DataFrame({
        'Seuquence ID' : [row_set] * len(market_subset),
        'End Date' : [close_date] * len(market_subset),
        'Close Gap' : close_gap,
        'High Gap' : high_gap,
        'Low Gap' : low_gap,
        'Volume Gap' : volume_gap,
        'Daily Change' : daily_change,
        'Next Day Price Movement' : close_price_next_day_direction                         
        }))
    
    counter += 1

new_row_set = pd.concat(new_row_set)
new_row_set = new_row_set.dropna(how='any')
new_row_set.to_csv('NRS.csv', index=False)