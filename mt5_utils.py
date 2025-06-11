# mt5_utils.py

import MetaTrader5 as mt5
import pandas as pd
import time
import numpy as np 
from datetime import datetime # Already added

# --- MT5 Connection and Data Retrieval ---
def connect_mt5(mt5_path: str, login: int, password: str, server: str, log_activity_func):
    """Connects to MetaTrader 5 terminal with retry logic."""
    max_retries = 5

    for attempt in range(max_retries):
        try:
            if not mt5.initialize(path=mt5_path):
                log_activity_func(f"Failed to initialize MT5 (Attempt {attempt + 1}/{max_retries})", "red")
                time.sleep(3)
                continue
                
            if not mt5.login(login=login, password=password, server=server):
                log_activity_func(f"Failed to login to MT5 (Attempt {attempt + 1}/{max_retries})", "red")
                mt5.shutdown()
                time.sleep(3)
                continue
                
            account_info = mt5.account_info()
            if account_info is None:
                log_activity_func(f"Failed to get account info (Attempt {attempt + 1}/{max_retries}). MT5 connection might be problematic.", "red")
                mt5.shutdown()
                time.sleep(3)
                continue
                
            log_activity_func("Connected successfully to MT5!", "green")
            return True
            
        except Exception as e:
            log_activity_func(f"Connection error: {str(e)}", "red")
            if attempt < max_retries - 1:
                log_activity_func("Retrying connection...", "yellow")
                time.sleep(3)
            else:
                return False
                
    return False

def get_live_data(symbol: str, timeframes_to_fetch: dict, num_candles: int, log_activity_func, calculate_indicators_func):
    """
    Fetches live market data for a given symbol across multiple specified timeframes.
    Returns a dictionary containing symbol info, current price, and indicator data
    for each requested timeframe.
    
    Args:
        symbol (str): The MT5 symbol name (e.g., "EURUSD").
        timeframes_to_fetch (dict): A dictionary mapping user-friendly TF names to mt5.TIMEFRAME_ constants.
        num_candles (int): Number of historical candles to fetch.
        log_activity_func (function): Function to log activities.
        calculate_indicators_func (function): Function from indicators.py to calculate indicators.

    Returns:
        dict: A dictionary with 'symbol', 'current_price', 'bid', 'ask', 'spread', 'volume',
              'time', 'all_timeframe_data', 'symbol_info', or None if data fetch fails.
    """
    all_timeframe_data = {}
    current_price = None
    spread = None
    tick_volume = None
    symbol_info_obj = None

    for timeframe_name, mt5_timeframe in timeframes_to_fetch.items():
        try:
            if current_price is None: 
                symbol_info_obj = mt5.symbol_info(symbol)
                if symbol_info_obj is None:
                    # log_activity_func(f"Could not get symbol info for {symbol}. Skipping.", "yellow")
                    return None
                
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    # log_activity_func(f"Could not get tick data for {symbol}. Skipping.", "yellow")
                    return None
                
                current_price = (tick.bid + tick.ask) / 2
                spread = tick.ask - tick.bid
                tick_volume = tick.volume

            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_candles)
            if rates is None or len(rates) == 0:
                # log_activity_func(f"Could not get historical rates for {symbol} on {timeframe_name}. Skipping this timeframe.", "yellow")
                continue
            
            df = pd.DataFrame(rates)
            if df.empty:
                continue
            
            df['time'] = pd.to_datetime(df['time'], unit='s')

            indicators_complete, latest_candle_data = calculate_indicators_func(df)

            all_timeframe_data[timeframe_name] = {
                'latest_candle_data': latest_candle_data,
                'indicators_complete': indicators_complete
            }

        except Exception as e:
            # log_activity_func(f"Error getting {timeframe_name} data for {symbol}: {str(e)}", "red")
            continue

    if not all_timeframe_data:
        return None

    return {
        'symbol': symbol,
        'current_price': current_price,
        'bid': tick.bid if 'tick' in locals() and tick is not None else np.nan,
        'ask': tick.ask if 'tick' in locals() and tick is not None else np.nan,
        'spread': spread,
        'volume': tick_volume,
        'time': datetime.now(),
        'all_timeframe_data': all_timeframe_data,
        'symbol_info': symbol_info_obj
    }

