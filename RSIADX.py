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

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import copy as c
from itertools import product
import BackTest_2 as BT

def get_all_dates(start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    return date_list

def rsi(symbol, window=14):
    df = c.deepcopy(symbol)
    df['T_1_close'] = df['close'].shift(1)
    df['max'] = df.apply(lambda x: max(x['close'] - x['T_1_close'], 0), axis=1)
    df['abs'] = df.apply(lambda x: abs(x['close'] - x['T_1_close']), axis=1)
    alpha = 2 / (1 + window)
    df['RSI'] = df['max'].ewm(min_periods=window, adjust=False, alpha=alpha).mean() / \
                df['abs'].ewm(min_periods=window, adjust=False, alpha=alpha).mean() * 100
    return df['RSI']

def adx(symbol, window=14):
    df = c.deepcopy(symbol)
    df['pDM'] = (df['high'] - df['high'].shift(1)).apply(lambda x: max(x, 0))
    df['nDM'] = (df['low'].shift(1) - df['low']).apply(lambda x: max(x, 0))
    df['TR'] = df.apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['close'] - x['low'])), axis=1)
    df['ATR'] = df['TR'].rolling(window).mean()
    df['pDI'] = df['pDM'].rolling(window).mean() / df['ATR'] * 100
    df['nDI'] = df['nDM'].rolling(window).mean() / df['ATR'] * 100
    df['ADX'] = ((df['pDI'] - df['nDI']).apply(abs) / (df['pDI'] + df['nDI'])).rolling(window).mean() * 100
    return df['ADX']

def getOption(close, df, td, maturity_date, level, CorP=True):
    CorP = 'call' if CorP else 'put'
    df_filtered = c.deepcopy(df[(df['timestamp'] == td) & (df['expiration'] == maturity_date) & (df['type'] == CorP)])
    df_filtered['underlying_close'] = close
    df_filtered['dis'] = abs(df_filtered['strike_price'] - df_filtered['underlying_close'])
    df_filtered.reset_index(drop=True, inplace=True)
    if len(df_filtered) == 0:
        return [np.nan, np.nan]
    index = df_filtered[df_filtered['dis'] == df_filtered['dis'].min()].index[0]
    if CorP == 'call':
        index = min(index + level, df_filtered.index[-1])
    else:
        index = max(index - level, df_filtered.index[0])
    return list(df_filtered.loc[index, ['open_interest', 'last_price']])

def generate_signals(df, rsi_low, rsi_high, rsi_very_high):
    return df.apply(lambda x: 1 if ((x['RSI'] >= rsi_low and x['RSI'] < rsi_high) or (x['RSI'] < rsi_low)) and (x['ADX'] >= x['pre_ADX']) else \
                    (-1 if (x['RSI'] >= rsi_very_high) and (x['ADX'] < x['pre_ADX']) else 0), axis=1)

def backtest(df, signals):
    Signal, Price, Direction, Deposit, flag, cover = [], [], [], [], False, True
    Volume, Position, Pct = [], [], []
    n = len(df)
    for i in range(n):
        if flag:
            if (i == n - 1) or (df['expiration'].iloc[i] != df['expiration'].iloc[i + 1]):  # Close position before expiration
                signal = -1
                price = [df['C_last_price'].iloc[i] if cover else df['P_last_price'].iloc[i]]
                direction = [-1]
                deposit = [df['C_open_interest'].iloc[i] if cover else df['P_open_interest'].iloc[i]]
                flag = False
            else:
                if cover:
                    if signals[i] == 1:
                        signal = -1
                        price = [df['C_last_price'].iloc[i]]
                        direction = [-1]
                        deposit = [df['C_open_interest'].iloc[i]]
                        flag = False
                    else:
                        signal = 0
                        price = [df['C_last_price'].iloc[i]]
                        direction = [0]
                        deposit = [df['C_open_interest'].iloc[i]]
                else:
                    if signals[i] == -1:
                        signal = -1
                        price = [df['P_last_price'].iloc[i]]
                        direction = [-1]
                        deposit = [df['P_open_interest'].iloc[i]]
                        flag = False
                    else:
                        signal = 0
                        price = [df['P_last_price'].iloc[i]]
                        direction = [0]
                        deposit = [df['P_open_interest'].iloc[i]]
        else:
            if signals[i] == -1:
                signal = 1
                price = [df['C_last_price'].iloc[i]]
                direction = [-1]
                deposit = [df['C_open_interest'].iloc[i]]
                flag, cover = True, True
            elif signals[i] == 1:
                signal = 1
                price = [df['P_last_price'].iloc[i]]
                direction = [-1]
                deposit = [df['P_open_interest'].iloc[i]]
                flag, cover = True, False
            else:
                signal = 0
                price = [0]
                direction = [0]
                deposit = [0]
        Signal.append(signal)
        Price.append(price)
        Direction.append(direction)
        Deposit.append(deposit)
        Volume.append([-1])
        Position.append([0])
        Pct.append([0])

    signalDf = pd.DataFrame({'timestamp': df['timestamp'], 'signal': Signal, 'price': Price, 'direction': Direction, 'volume': Volume, 'deposit': Deposit, 'position': Position, 'pct': Pct})
    signalDf['trade_date'] = signalDf['timestamp']
    
    try:
        recordDf = BT.OptionBT(signalDf)
        return recordDf.capital.values[-1]
    except Exception as e:
        print(f"Error in OptionBT: {e}")
        return float('-inf')  # 返回一个非常小的值表示这组参数无效

def hyperparameter_search(df):
    rsi_low_range = range(10, 31, 5)
    rsi_high_range = range(50, 71, 5)
    rsi_very_high_range = range(70, 91, 5)
    
    best_params = None
    best_performance = float('-inf')
    
    for rsi_low, rsi_high, rsi_very_high in product(rsi_low_range, rsi_high_range, rsi_very_high_range):
        if rsi_low < rsi_high < rsi_very_high:
            signals = generate_signals(df, rsi_low, rsi_high, rsi_very_high)
            performance = backtest(df, signals)
            
            if performance > best_performance:
                best_performance = performance
                best_params = (rsi_low, rsi_high, rsi_very_high)
            
            print(f"RSI Low: {rsi_low}, RSI High: {rsi_high}, RSI Very High: {rsi_very_high}, Performance: {performance}")
    
    return best_params, best_performance

# Main execution
start_date = "2019-03-30"
end_date = "2019-05-27"
all_dates = get_all_dates(start_date, end_date)
price_data = pd.read_csv(f'D:/datasets/{underlying}_price_data.csv')
for c in ['open', 'high', 'low', 'close', 'volume',]:
    price_data[c]=price_data[c].shift(1).values.flatten()
# Load and preprocess option data
df = pd.concat([pd.read_csv(f'D:/datasets/BTC{date}') for date in all_dates[:-1]], axis=0)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['expiration'] = pd.to_datetime(df['expiration'])
df.index = range(len(df))
df=pd.merge(df, price_data, on='date', how='left')

# Calculate technical indicators
df['RSI'] = rsi(df)
df['ADX'] = adx(df)
df['pre_ADX'] = df['ADX'].shift(1)
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

# Option Selection
df['option'] = df.apply(lambda x: getOption(x['close'], df, x['timestamp'], x['expiration'], 0, True), axis=1)
df[['C_open_interest', 'C_last_price']] = df['option'].apply(pd.Series)
df['option'] = df.apply(lambda x: getOption(x['close'], df, x['timestamp'], x['expiration'], 0, False), axis=1)
df[['P_open_interest', 'P_last_price']] = df['option'].apply(pd.Series)

# Perform hyperparameter search
best_params, best_performance = hyperparameter_search(df)

print(f"Best parameters: RSI Low: {best_params[0]}, RSI High: {best_params[1]}, RSI Very High: {best_params[2]}")
print(f"Best performance: {best_performance}")

# Generate final signals with best parameters
final_signals = generate_signals(df, best_params[0], best_params[1], best_params[2])

# Perform final backtest with best parameters
final_performance = backtest(df, final_signals)
print(f"Final performance with best parameters: {final_performance}")
