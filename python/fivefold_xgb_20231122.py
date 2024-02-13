import gc
import os
import time
import warnings
from itertools import combinations
from warnings import simplefilter
import xgboost as xgb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit

warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

is_offline = False
is_train = True
is_infer = True
max_lookback = np.nan
split_day = 435

df = pd.read_csv("train.csv")
df = df.dropna(subset=["target"])
df.reset_index(drop=True, inplace=True)
df.shape

def reduce_mem_usage(df, verbose=0):
    """
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    if verbose:
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Decreased by {decrease:.2f}%")

    return df

from numba import njit, prange

@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            if mid_val == min_val:  # Prevent division by zero
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features


def calculate_triplet_imbalance_numba(price, df):
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]

    # Calculate the triplet imbalance
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)

    return features
####Test
# @njit(parallel = True)
# def calculate_rsi(prices, period=14):
#     rsi_values = np.zeros_like(prices)

#     for col in prange(prices.shape[1]):
#         price_data = prices[:, col]
#         delta = np.zeros_like(price_data)
#         delta[1:] = price_data[1:] - price_data[:-1]
#         gain = np.where(delta > 0, delta, 0)
#         loss = np.where(delta < 0, -delta, 0)

#         avg_gain = np.mean(gain[:period])
#         avg_loss = np.mean(loss[:period])
        
#         if avg_loss != 0:
#             rs = avg_gain / avg_loss
#         else:
#             rs = 1e-9  # or any other appropriate default value
            
#         rsi_values[:period, col] = 100 - (100 / (1 + rs))

#         for i in prange(period-1, len(price_data)-1):
#             avg_gain = (avg_gain * (period - 1) + gain[i]) / period
#             avg_loss = (avg_loss * (period - 1) + loss[i]) / period
#             if avg_loss != 0:
#                 rs = avg_gain / avg_loss
#             else:
#                 rs = 1e-9  # or any other appropriate default value
#             rsi_values[i+1, col] = 100 - (100 / (1 + rs))

#     return rsi_values
# @njit(parallel=True)
# def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
#     rows, cols = data.shape
#     macd_values = np.empty((rows, cols))
#     signal_line_values = np.empty((rows, cols))
#     histogram_values = np.empty((rows, cols))

#     for i in prange(cols):
#         short_ema = np.zeros(rows)
#         long_ema = np.zeros(rows)

#         for j in range(1, rows):
#             short_ema[j] = (data[j, i] - short_ema[j - 1]) * (2 / (short_window + 1)) + short_ema[j - 1]
#             long_ema[j] = (data[j, i] - long_ema[j - 1]) * (2 / (long_window + 1)) + long_ema[j - 1]

#         macd_values[:, i] = short_ema - long_ema

#         signal_line = np.zeros(rows)
#         for j in range(1, rows):
#             signal_line[j] = (macd_values[j, i] - signal_line[j - 1]) * (2 / (signal_window + 1)) + signal_line[j - 1]

#         signal_line_values[:, i] = signal_line
#         histogram_values[:, i] = macd_values[:, i] - signal_line

#     return macd_values, signal_line_values, histogram_values

# @njit(parallel=True)
# def calculate_bband(data, window=20, num_std_dev=2):
#     num_rows, num_cols = data.shape
#     upper_bands = np.zeros_like(data)
#     lower_bands = np.zeros_like(data)
#     mid_bands = np.zeros_like(data)

#     for col in prange(num_cols):
#         for i in prange(window - 1, num_rows):
#             window_slice = data[i - window + 1 : i + 1, col]
#             mid_bands[i, col] = np.mean(window_slice)
#             std_dev = np.std(window_slice)
#             upper_bands[i, col] = mid_bands[i, col] + num_std_dev * std_dev
#             lower_bands[i, col] = mid_bands[i, col] - num_std_dev * std_dev

#     return upper_bands, mid_bands, lower_bands

# def generate_ta(df):
#     # Define lists of price and size-related column names
#     prices = ["reference_price","ask_price", "bid_price", "wap"]
#     # sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    
#     for stock_id, values in df.groupby(['stock_id'])[prices]:
#         # RSI
#         col_rsi = [f'rsi_{col}' for col in values.columns]
#         rsi_values = calculate_rsi(values.values)
#         df.loc[values.index, col_rsi] = rsi_values
#         gc.collect()
        
#         # MACD
#         macd_values, signal_line_values, histogram_values = calculate_macd(values.values)
#         col_macd = [f'macd_{col}' for col in values.columns]
#         col_signal = [f'macd_sig_{col}' for col in values.columns]
#         col_hist = [f'macd_hist_{col}' for col in values.columns]
        
#         df.loc[values.index, col_macd] = macd_values
#         df.loc[values.index, col_signal] = signal_line_values
#         df.loc[values.index, col_hist] = histogram_values
#         gc.collect()
        
#         # Bollinger Bands
#         bband_upper_values, bband_mid_values, bband_lower_values = calculate_bband(values.values, window=20, num_std_dev=2)
#         col_bband_upper = [f'bband_upper_{col}' for col in values.columns]
#         col_bband_mid = [f'bband_mid_{col}' for col in values.columns]
#         col_bband_lower = [f'bband_lower_{col}' for col in values.columns]
        
#         df.loc[values.index, col_bband_upper] = bband_upper_values
#         df.loc[values.index, col_bband_mid] = bband_mid_values
#         df.loc[values.index, col_bband_lower] = bband_lower_values
#         gc.collect()
    
#     return df

####Test

# generate imbalance features
def imbalance_features(df):
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    # V1
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["size_imbalance"] = df.eval("bid_size / ask_size")
    
    df["imb_new"] = df["imbalance_size"] * df['imbalance_buy_sell_flag']
    df["matched_size_new"] = df["matched_size"] * df['imbalance_buy_sell_flag']
    df["matched_imbalance_3"] = df.eval("imb_new/matched_size")
    
    df['median_sizes_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['bid_size'].transform('median') + df.groupby(['seconds_in_bucket', 'date_id'])['ask_size'].transform('median')
    df['std_sizes_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['bid_size'].transform('std') + df.groupby(['seconds_in_bucket', 'date_id'])['ask_size'].transform('std')
    df['max_sizes_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['bid_size'].transform('max') + df.groupby(['seconds_in_bucket', 'date_id'])['ask_size'].transform('max')
    df['min_sizes_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['bid_size'].transform('min') + df.groupby(['seconds_in_bucket', 'date_id'])['ask_size'].transform('min')
    df['mean_sizes_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['bid_size'].transform('mean') + df.groupby(['seconds_in_bucket', 'date_id'])['ask_size'].transform('mean')
    
    df['median_imb_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['size_imbalance'].transform('median') 
    df['std_imb_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['size_imbalance'].transform('std') 
    df['max_imb_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['size_imbalance'].transform('max') 
    df['min_imb_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['size_imbalance'].transform('min') 
    df['mean_imb_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['size_imbalance'].transform('mean')
    
    df['median_imb2_t'] = df.groupby(['seconds_in_bucket', 'date_id'])["wap"].transform('median') 
    df['std_imb2_t'] = df.groupby(['seconds_in_bucket', 'date_id'])["wap"].transform('std') 
    df['max_imb2_t'] = df.groupby(['seconds_in_bucket', 'date_id'])["wap"].transform('max') 
    df['min_imb2_t'] = df.groupby(['seconds_in_bucket', 'date_id'])["wap"].transform('min') 
    df['mean_imb2_t'] = df.groupby(['seconds_in_bucket', 'date_id'])["wap"].transform('mean')
    
    df["range_sizes_t"] = df['max_sizes_t'] - df['min_sizes_t']
    df["range_imb_t"] = df['max_imb_t'] - df['min_imb_t']
    df["range_imb2_t"] = df["max_imb2_t"] - df["min_imb2_t"]
    
    # df['median_sizes_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['volume'].transform('median') 
    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    for c in [['ask_price', 'bid_price', 'wap', 'reference_price']]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values
        
    # V2
    df["stock_weights"] = df["stock_id"].map(weights)
    df["weighted_wap"] = df["stock_weights"] * df["wap"]
    df['wap_momentum'] = df.groupby('stock_id')['weighted_wap'].pct_change(periods=6)
    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imb_new'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    # df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    df['spread_depth_ratio'] = (df['ask_price'] - df['bid_price']) / (df['bid_size'] + df['ask_size'])
    #df['mid_price_movement'] = df['mid_price'].diff(periods=5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df['micro_price'] = ((df['bid_price'] * df['ask_size']) + (df['ask_price'] * df['bid_size'])) / (df['bid_size'] + df['ask_size'])
    df['relative_spread'] = (df['ask_price'] - df['bid_price']) / df['wap']
    
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
    
    # df['status'] = df['seconds_in_bucket'].apply(lambda x: 1 if x<300 else 2 if x<480 else 3)
    # V3
    for col in ['matched_size_new', 'imb_new', 'reference_price',"matched_imbalance_3","reference_price_wap_imb"]:
        for window in [1, 2, 3, 5, 10]:
            df[f"{col}_shift_{window}"] = df.groupby(['stock_id','date_id'])[col].shift(window)
            
    for col in ['matched_size_new', 'imb_new', 'reference_price',"matched_imbalance_3"]:
        for window in [1, 2, 3, 5, 10]:
            df[f"{col}_ret_{window}"] = df.groupby(['stock_id','date_id'])[col].pct_change(window)
    
    for col in ['ask_price', 'bid_price',
                'wap', 'near_price', 'far_price',"reference_price_wap_imb","market_urgency","imb_new"]:
        for window in [1, 2, 3, 5, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)
    df['median_weighted_wap_t'] = df.groupby(['seconds_in_bucket', 'date_id'])["weighted_wap"].transform('median') 
    df['std_weighted_wap_t'] = df.groupby(['seconds_in_bucket', 'date_id'])["weighted_wap"].transform('std') 
    df['max_weighted_wap_t'] = df.groupby(['seconds_in_bucket', 'date_id'])["weighted_wap"].transform('max') 
    df['mean_weighted_wap_t'] = df.groupby(['seconds_in_bucket', 'date_id'])["weighted_wap"].transform('mean')
    df['median_market_urgency_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['market_urgency'].transform('median') 
    df['std_market_urgency_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['market_urgency'].transform('std') 
    df['max_market_urgency_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['market_urgency'].transform('max') 
    df['min_market_urgency_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['market_urgency'].transform('min') 
    df['mean_market_urgency_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['market_urgency'].transform('mean')
    df["range_market_urgency_t"] = df["max_market_urgency_t"] - df["min_market_urgency_t"]
    df['median_reference_price_wap_imb_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['reference_price_wap_imb'].transform('median') 
    df['std_reference_price_wap_imb_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['reference_price_wap_imb'].transform('std') 
    df['max_reference_price_wap_imb_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['reference_price_wap_imb'].transform('max') 
    df['min_reference_price_wap_imb_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['reference_price_wap_imb'].transform('min') 
    df['mean_reference_price_wap_imb_t'] = df.groupby(['seconds_in_bucket', 'date_id'])['reference_price_wap_imb'].transform('mean')
    df["range_reference_price_wap_imb_t"] = df["max_reference_price_wap_imb_t"] - df["min_reference_price_wap_imb_t"]
    
    #df['imb_flag_aggregate1'] = df.groupby(['stock_id',"date_id"])['imbalance_buy_sell_flag'].rolling(window=1).mean().reset_index(level=(0,1), drop=True).sort_index()
    #df['imb_flag_aggregate2'] = df.groupby(['stock_id',"date_id"])['imbalance_buy_sell_flag'].rolling(window=2).mean().reset_index(level=(0,1), drop=True).sort_index()
    #df['imb_flag_aggregate3'] = df.groupby(['stock_id',"date_id"])['imbalance_buy_sell_flag'].rolling(window=3).mean().reset_index(level=(0,1), drop=True).sort_index()
    #df['imb_flag_aggregate5'] = df.groupby(['stock_id',"date_id"])['imbalance_buy_sell_flag'].rolling(window=5).mean().reset_index(level=(0,1), drop=True).sort_index()
    df['wap_rolling_mean5'] = df.groupby(['stock_id',"date_id"])['wap'].rolling(window=5).mean().reset_index(level=(0,1), drop=True).sort_index()
    df['wap_rolling_mean10'] = df.groupby(['stock_id',"date_id"])['wap'].rolling(window=10).mean().reset_index(level=(0,1), drop=True).sort_index()
    df['wap_rolling_mean15'] = df.groupby(['stock_id',"date_id"])['wap'].rolling(window=15).mean().reset_index(level=(0,1), drop=True).sort_index()
    df['wap_rolling_std5'] = df.groupby(['stock_id',"date_id"])['wap'].rolling(window=5).std().reset_index(level=(0,1), drop=True).sort_index()
    df['wap_rolling_std10'] = df.groupby(['stock_id',"date_id"])['wap'].rolling(window=10).std().reset_index(level=(0,1), drop=True).sort_index()
    df['wap_rolling_std15'] = df.groupby(['stock_id',"date_id"])['wap'].rolling(window=15).std().reset_index(level=(0,1), drop=True).sort_index()
    df['wap_rolling_ptp5'] = df.groupby(['stock_id',"date_id"])['wap'].rolling(window=5).max().reset_index(level=(0,1), drop=True).sort_index() -  df.groupby(['stock_id',"date_id"])['wap'].rolling(window=5).min().reset_index(level=(0,1), drop=True).sort_index()
    df['wap_rolling_ptp10'] = df.groupby(['stock_id',"date_id"])['wap'].rolling(window=10).max().reset_index(level=(0,1), drop=True).sort_index() -  df.groupby(['stock_id',"date_id"])['wap'].rolling(window=10).min().reset_index(level=(0,1), drop=True).sort_index()
    df['wap_rolling_ptp15'] = df.groupby(['stock_id',"date_id"])['wap'].rolling(window=15).max().reset_index(level=(0,1), drop=True).sort_index() -  df.groupby(['stock_id',"date_id"])['wap'].rolling(window=15).min().reset_index(level=(0,1), drop=True).sort_index()
    
    
    df['bid_size_rolling_mean5'] = df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=5).mean().reset_index(level=(0,1), drop=True).sort_index()
    df['bid_size_rolling_mean10'] = df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=10).mean().reset_index(level=(0,1), drop=True).sort_index()
    df['bid_size_rolling_mean15'] = df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=10).mean().reset_index(level=(0,1), drop=True).sort_index()
    df['bid_size_rolling_std5'] = df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=5).std().reset_index(level=(0,1), drop=True).sort_index()
    df['bid_size_rolling_std10'] = df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=10).std().reset_index(level=(0,1), drop=True).sort_index()
    df['bid_size_rolling_std15'] = df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=15).std().reset_index(level=(0,1), drop=True).sort_index()
    df['bid_size_rolling_ptp5'] = df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=5).max().reset_index(level=(0,1), drop=True).sort_index() -  df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=5).min().reset_index(level=(0,1), drop=True).sort_index()
    df['bid_size_rolling_ptp10'] = df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=10).max().reset_index(level=(0,1), drop=True).sort_index() -  df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=10).min().reset_index(level=(0,1), drop=True).sort_index()
    df['bid_size_rolling_ptp15'] = df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=15).max().reset_index(level=(0,1), drop=True).sort_index() -  df.groupby(['stock_id',"date_id"])['bid_size'].rolling(window=15).min().reset_index(level=(0,1), drop=True).sort_index()
    
    
    #df['imb_flag_aggregate10'] = df.groupby(['stock_id',"date_id"])['imbalance_buy_sell_flag'].rolling(window=10).mean().reset_index(level=(0,1), drop=True).sort_index()
    return df.replace([np.inf, -np.inf], 0)

# generate time & stock features
def other_features(df):
    df["dow"] = df["date_id"] % 5
    df["dom"] = df["date_id"] % 20
    # df["seconds"] = df["seconds_in_bucket"] % 60
    df["minute"] = df["seconds_in_bucket"] // 60

    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df

# generate all features
def generate_all_features(df):
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    df = imbalance_features(df)
    df = other_features(df)
    gc.collect()
    
    feature_name = [i for i in df.columns if i not in ["imbalance_buy_sell_flag","stock_weights","row_id", "target", "time_id", "date_id","far_price_diff_2","matched_imbalance_ret_2","matched_imbalance_ret_3","bid_size_ask_size_imbalance_size_imb2","imbalance_buy_sell_flag_shift_5"]]
    
    return df[feature_name]

weights = [
    0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
    0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
    0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
    0.004, 0.004, 0.006, 0.002, 0.002, 0.04 , 0.002, 0.002, 0.004, 0.04 , 0.002, 0.001,
    0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
    0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
    0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
    0.02 , 0.004, 0.006, 0.002, 0.02 , 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
    0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
    0.004, 0.006, 0.006, 0.001, 0.04 , 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
    0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
    0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
    0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
    0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
    0.04 , 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02 , 0.004, 0.002, 0.006, 0.02,
    0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
    0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
]

weights = {int(k):v for k,v in enumerate(weights)}

if is_offline:
    df_train = df[df["date_id"] <= split_day]
    df_valid = df[df["date_id"] > split_day]
    print("Offline mode")
    print(f"train : {df_train.shape}, valid : {df_valid.shape}")
else:
    df_train = df
    print("Online mode")

df_train['market_urgency'] = df_train.eval("(ask_price-bid_price)*(bid_size-ask_size)/(bid_size+ask_size)")
df_train["size_imbalance"] = df_train.eval("bid_size / ask_size")

if is_train:
    global_stock_id_feats = {
        "median_size": df_train.groupby("stock_id")["bid_size"].median() + df_train.groupby("stock_id")["ask_size"].median(),
        "std_size": df_train.groupby("stock_id")["bid_size"].std() + df_train.groupby("stock_id")["ask_size"].std(),
        "ptp_size": df_train.groupby("stock_id")["bid_size"].max() - df_train.groupby("stock_id")["bid_size"].min(),
        "median_price": df_train.groupby("stock_id")["bid_price"].median() + df_train.groupby("stock_id")["ask_price"].median(),
        "std_price": df_train.groupby("stock_id")["bid_price"].std() + df_train.groupby("stock_id")["ask_price"].std(),
        "ptp_price": df_train.groupby("stock_id")["bid_price"].max() - df_train.groupby("stock_id")["ask_price"].min(),
        "std_wap": df_train.groupby("stock_id")["wap"].std(),
        "median_wap": df_train.groupby("stock_id")["wap"].median(),
        "ptp_wap": df_train.groupby("stock_id")["wap"].max() - df_train.groupby("stock_id")["wap"].min(),
        "std_market_urgency": df_train.groupby("stock_id")["market_urgency"].std(),
        "median_market_urgency": df_train.groupby("stock_id")["market_urgency"].median(),
        "ptp_market_urgency": df_train.groupby("stock_id")["market_urgency"].max() - df_train.groupby("stock_id")["market_urgency"].min(),
        "std_size_imbalance": df_train.groupby("stock_id")["size_imbalance"].std(),
        "median_size_imbalance": df_train.groupby("stock_id")["size_imbalance"].median(),
        "ptp_size_imbalance": df_train.groupby("stock_id")["size_imbalance"].max() - df_train.groupby("stock_id")["size_imbalance"].min(),
    }
    if is_offline:
        df_train_feats = generate_all_features(df_train)
        print("Build Train Feats Finished.")
        df_valid_feats = generate_all_features(df_valid)
        print("Build Valid Feats Finished.")
        df_valid_feats = reduce_mem_usage(df_valid_feats)
    else:
        df_train_feats = generate_all_features(df_train)
        print("Build Online Train Feats Finished.")

    df_train_feats = reduce_mem_usage(df_train_feats)
    
if is_train:
    feature_name = list(df_train_feats.columns)
    for nl in [1200]:
        for ss in [0.55]:
            for cb in [0.45]:
                print((nl,ss,cb))
                xgb_params = {
                    "objective": "reg:absoluteerror",
                    "eval_metric":"mae",
                    "n_estimators": 30000,
                    "max_depth":11,
                    "subsample":ss,
                    "colsample_bytree": cb,
                    "learning_rate": 0.004,
                    "n_jobs": 10,
                    "device":"gpu",
                }

                for i in range(5):
                    print(f"Feature length = {len(feature_name)}")

                    offline_split = (df_train['date_id']>= 96*i ) & (df_train['date_id']<= 96*(i+1) )
                    df_offline_train = df_train_feats[~offline_split]
                    df_offline_valid = df_train_feats[offline_split]
                    df_offline_train_target = df_train['target'][~offline_split]
                    df_offline_valid_target = df_train['target'][offline_split]

                    print("Valid Model Trainning.")
                    lgb_model = xgb.XGBRegressor(**xgb_params)
                    lgb_model.fit(
                    df_offline_train[feature_name],
                    df_offline_train_target,
                    eval_set=[(df_offline_valid[feature_name], df_offline_valid_target)],
                    early_stopping_rounds=1500,
                    verbose=True
                    )
                    os.system('mkdir models_xgb_fivefold'+str(len(feature_name)))
                    joblib.dump(lgb_model, './models_xgb_fivefold'+str(len(feature_name))+f'/xgbPart_{i}.model')
                    pd.options.display.max_rows = 4000
                    b =pd.DataFrame(lgb_model.feature_importances_,index=df_train_feats.columns)
                    print(b.sort_values(by=0,ascending=False))

    del df_offline_train, df_offline_valid, df_offline_train_target, df_offline_valid_target
    gc.collect()

    # infer
    df_train_target = df_train["target"]
    print("Infer Model Trainning.")
    infer_params = xgb_params.copy()
    infer_params["n_estimators"] = int(1.8 * lgb_model.best_iteration)
    infer_lgb_model = xgb.XGBRegressor(**infer_params)
    infer_lgb_model.fit(df_train_feats[feature_name], df_train_target)
    joblib.dump(infer_lgb_model, './models_xgb_fivefold'+str(len(feature_name))+'/xgbFull.model')
    if is_offline:   
        # offline predictions
        df_valid_target = df_valid["target"]
        offline_predictions = infer_lgb_model.predict(df_valid_feats[feature_name])
        offline_score = mean_absolute_error(offline_predictions, df_valid_target)
        print(f"Offline Score {np.round(offline_score, 4)}")