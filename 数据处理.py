import pandas as pd
import requests
import time
from datetime import datetime, timedelta

def get_binance_klines(symbol, interval, start_time, end_time):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}"
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                     'close_time', 'quote_asset_volume', 'number_of_trades',
                                     'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df.astype(float)
    
    return df

def fetch_binance_data(symbol, start_date, end_date):
    start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)
    
    interval = '1d'  # Daily data
    
    price_data = pd.DataFrame()
    
    while start_timestamp < end_timestamp:
        print(f"Fetching data from {pd.to_datetime(start_timestamp, unit='ms')} to {pd.to_datetime(end_timestamp, unit='ms')}")
        temp_data = get_binance_klines(symbol, interval, start_timestamp, end_timestamp)
        price_data = price_data.append(temp_data)
        
        start_timestamp = int(temp_data.index[-1].timestamp() * 1000) + 1
        time.sleep(1)  # Pause for 1 second to avoid exceeding API request limits
    
    return price_data

# Example usage
symbol = "BTCUSDT"
start_date = "2019-01-01"
end_date = "2019-12-31"
underlying = "BTC"

price_data = fetch_binance_data(symbol, start_date, end_date)
price_data['date'] = price_data.index
price_data.to_csv(f'D:/datasets/{underlying}_price_data.csv',index=False)

import numpy as np
import talib

def get_all_dates(start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    return date_list

def select_option(options_data, underlying_price, days_to_expiry_threshold=7):
    # 将时间戳转换为日期时间
    options_data['timestamp'] = pd.to_datetime(options_data['timestamp'], unit='us')
    options_data['expiration'] = pd.to_datetime(options_data['expiration'], unit='us')
    
    # 计算到期日
    current_date = options_data['timestamp'].max().normalize()
    options_data['days_to_expiration'] = (options_data['expiration'] - current_date).dt.days
    
    # 根据到期日过滤期权
    minexpire=options_data['days_to_expiration'].min()
    if minexpire < days_to_expiry_threshold:
        options_data = options_data[options_data['days_to_expiration'] >= days_to_expiry_threshold]
    minexpire=options_data['days_to_expiration'].min()
    options_data = options_data[options_data['days_to_expiration'] == minexpire]
    # Select at-the-money options for both call and put
    atm_call_options = options_data[options_data['type'] == 'call'].loc[
        [options_data[options_data['type'] == 'call'].groupby('type')['strike_price'].apply(lambda x: abs(x - underlying_price)).idxmin()]
    ]
    atm_put_options = options_data[options_data['type'] == 'put'].loc[
        [options_data[options_data['type'] == 'put'].groupby('type')['strike_price'].apply(lambda x: abs(x - underlying_price)).idxmin()]
    ]
    
    # 连接结果
    atm_options = pd.concat([atm_call_options, atm_put_options], axis=0)
    
    return atm_options

def calculate_signals(price_data):
    # Example signal calculation using TA-Lib
    price_data['SMA'] = talib.SMA(price_data['close'], timeperiod=20)
    price_data['RSI'] = talib.RSI(price_data['close'], timeperiod=14)
    return price_data

# Example usage
start_date = "2019-03-30"
end_date = "2019-05-28"
all_dates = get_all_dates(start_date, end_date)
underlying = 'BTC'

for date in all_dates[:]:
    print(f"Processing {date}")

    # Load price data (assuming you have a separate CSV for price data)
    price_data = pd.read_csv(f'D:/datasets/{underlying}_price_data.csv')

    # Load options data
    options_data = pd.read_csv(f'D:/datasets/deribit_options_chain_{date}_OPTIONS.csv')
    options_data = options_data[options_data['symbol'].str.contains(underlying)]

    # Get underlying price (assuming it's the opening price of the current day)
    underlying_price = price_data.loc[price_data['date'] == date, 'open'].values[0]
    
    # Select option
    selected_option = select_option(options_data, underlying_price)
    
    # Ensure liquidity, bid-ask spread, and implied volatility checks
    selected_option['bid_ask_spread'] = selected_option['ask_price'] - selected_option['bid_price']
    """postselected_option = selected_option[(selected_option['open_interest'] > 10) & 
                                      (selected_option['bid_ask_spread'] < 0.05) & 
                                      (selected_option['mark_iv'] < 1.0)]"""
    selected_option['date']=date
    selected_option.drop(['local_timestamp'],axis=1)
    selected_option.to_csv(f'D:/datasets/BTC{date}',index=False)
