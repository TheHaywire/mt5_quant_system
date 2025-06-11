# indicators.py

import pandas as pd
import pandas_ta as ta
import numpy as np 

def calculate_indicators(df: pd.DataFrame):
    """
    Calculates various technical indicators for a given DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data (open, high, low, close, tick_volume).
        
    Returns:
        tuple: (indicators_complete_status: bool, latest_candle_data_dict: dict)
               True if enough data for all indicators, False otherwise.
               latest_candle_data_dict contains indicator values for the last candle.
    """
    if df.empty:
        return False, {}

    min_candles_needed = max(200, 14) 

    required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan 

    if len(df) < min_candles_needed:
        return False, df.iloc[-1].to_dict() if not df.empty else {}
    
    df['ema_fast'] = ta.ema(df['close'], length=8)
    df['ema_slow'] = ta.ema(df['close'], length=21)
    
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['sma_200'] = ta.sma(df['close'], length=200)
    
    adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_data is not None and not adx_data.empty and 'ADX_14' in adx_data.columns:
        df['adx'] = adx_data['ADX_14']
        df['dmp'] = adx_data['DMP_14'] 
        df['dmn'] = adx_data['DMN_14'] 
    else:
        df['adx'] = np.nan
        df['dmp'] = np.nan
        df['dmn'] = np.nan

    df['rsi'] = ta.rsi(df['close'], length=14)
    
    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty and 'MACD_12_26_9' in macd_df.columns:
        df['macd'] = macd_df['MACD_12_26_9']
        df['macd_signal'] = macd_df['MACDs_12_26_9']
        df['macd_hist'] = macd_df['MACDh_12_26_9']
    else:
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14) 
    
    bbands_df = ta.bbands(df['close'], length=20)
    if bbands_df is not None and not bbands_df.empty and len(bbands_df.columns) >= 3:
        bb_cols = [col for col in bbands_df.columns if 'BBL' in col or 'BBM' in col or 'BBU' in col]
        if len(bb_cols) >= 3:
            df['bb_lower'] = bbands_df[bb_cols[0]] if 'BBL' in bb_cols[0] else bbands_df[bb_cols[2]]
            df['bb_middle'] = bbands_df[bb_cols[1]]
            df['bb_upper'] = bbands_df[bb_cols[2]] if 'BBU' in bb_cols[2] else bbands_df[bb_cols[0]]
        else:
            df['bb_upper'] = np.nan
            df['bb_middle'] = np.nan
            df['bb_lower'] = np.nan
    else:
        df['bb_upper'] = np.nan
        df['bb_middle'] = np.nan
        df['bb_lower'] = np.nan

    if 'tick_volume' in df.columns and not df['tick_volume'].isnull().all():
        df['volume_ma'] = df['tick_volume'].rolling(20).mean()
        df['volume_ratio'] = df.apply(lambda row: row['tick_volume'] / row['volume_ma'] if row['volume_ma'] != 0 else np.nan, axis=1)
    else:
        df['volume_ma'] = np.nan
        df['volume_ratio'] = np.nan

    df['body'] = df['close'] - df['open']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_size'] = abs(df['body'])
    df['candle_size'] = df['high'] - df['low'] 
    
    return True, df.iloc[-1].to_dict()

