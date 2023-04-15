import yfinance as yf
import pandas as pd
import numpy as np


def create_dataset(ticker,csv_address):
    snp = yf.Ticker("SWPPX")
    stock = yf.Ticker(ticker)
    stock_history = stock.history('max')
    snp_history = snp.history('max')

    snp_history.index = snp_history.index.tz_localize(None)
    stock_history.index = stock_history.index.tz_localize(None)

    first_stock_day = stock_history.index[0]
    first_snp_day = snp_history.index[0]

    last_stock_day = stock_history.index[-1]
    last_snp_day = snp_history.index[-1]

    if(first_stock_day>first_snp_day):
        stock_start = stock_history.head(1).index.strftime("%Y-%m-%d")[0]
        snp_history = snp_history[stock_start:]
    elif(first_stock_day<first_snp_day):
        snp_start = snp_history.head(1).index.strftime("%Y-%m-%d")[0]
        stock_history = stock_history[snp_start:]

    if(last_stock_day>last_snp_day):
        snp_end = snp_history.tail(1).index.strftime("%Y-%m-%d")[0]
        stock_history = stock_history[:snp_end]
    elif(last_stock_day<last_snp_day):
        stock_end = stock_history.tail(1).index.strftime("%Y-%m-%d")[0]
        snp_history = snp_history[:stock_end]

    snp_history = snp_history.rename(columns={'Open':'snp_Open', 'High':'snp_High', 'Low':'snp_Low', 'Close':'snp_Close', 'Volume':'snp_Volume', 'Dividends':'snp_Dividends', 'Stock Splits':'snp_Stock Splits'})
    result = pd.concat([stock_history,snp_history],axis=1)
    result.to_csv(csv_address)
    print (result.shape[1])

if __name__ == "__main__":
    ticker = 'MCD'
    save_to = './Data/Learning Data/snp_mcd_fullscope_daily.csv'
    create_dataset(ticker,save_to)
    