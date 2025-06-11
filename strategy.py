# strategy.py

import pandas as pd
import numpy as np 
from rich.text import Text 

# Helper to safely retrieve indicator values, defaulting to NaN if not present
def _get_val(data_dict: dict, key: str, default=np.nan):
    """Safely retrieves a value from a dictionary, handling NaN."""
    val = data_dict.get(key, default)
    return val if not pd.isna(val) else np.nan

def get_signal_strength(latest_data_dict: dict, config_adx_threshold: int):
    """
    Calculates buy/sell signal strength based on the latest indicators
    from a single timeframe's candle data.
    
    Args:
        latest_data_dict (dict): Dictionary of latest indicator values for a single candle.
        config_adx_threshold (int): The ADX threshold from CONFIG.
        
    Returns:
        tuple: (buy_strength: float, sell_strength: float)
               Scores between 0 and 1.
    """
    buy_conditions = []
    sell_conditions = []

    # Retrieve indicator values using helper
    rsi = _get_val(latest_data_dict, 'rsi')
    ema_fast = _get_val(latest_data_dict, 'ema_fast')
    ema_slow = _get_val(latest_data_dict, 'ema_slow')
    macd = _get_val(latest_data_dict, 'macd')
    macd_signal = _get_val(latest_data_dict, 'macd_signal')
    macd_hist = _get_val(latest_data_dict, 'macd_hist')
    volume_ratio = _get_val(latest_data_dict, 'volume_ratio')
    close_price = _get_val(latest_data_dict, 'close')
    open_price = _get_val(latest_data_dict, 'open')
    sma_50 = _get_val(latest_data_dict, 'sma_50')
    sma_200 = _get_val(latest_data_dict, 'sma_200')
    bb_upper = _get_val(latest_data_dict, 'bb_upper')
    bb_lower = _get_val(latest_data_dict, 'bb_lower')
    adx = _get_val(latest_data_dict, 'adx')
    dmp = _get_val(latest_data_dict, 'dmp') # +DI
    dmn = _get_val(latest_data_dict, 'dmn') # -DI

    # Calculate differences/ratios
    ema_diff = ema_fast - ema_slow if not (pd.isna(ema_fast) or pd.isna(ema_slow)) else np.nan
    body = close_price - open_price if not (pd.isna(close_price) or pd.isna(open_price)) else np.nan

    bb_position = np.nan
    if not (pd.isna(close_price) or pd.isna(bb_upper) or pd.isna(bb_lower)) and (bb_upper - bb_lower) != 0:
        bb_position = (close_price - bb_lower) / (bb_upper - bb_lower)

    # --- Buy Conditions ---
    if not pd.isna(rsi) and rsi < 30: buy_conditions.append(True) 
    if not pd.isna(ema_diff) and ema_diff > 0: buy_conditions.append(True) 
    if not (pd.isna(macd) or pd.isna(macd_signal)) and macd > macd_signal and macd_hist > 0: buy_conditions.append(True) 
    if not pd.isna(volume_ratio) and volume_ratio > 1.2: buy_conditions.append(True) 
    if not pd.isna(body) and body > 0: buy_conditions.append(True) 
    if not pd.isna(close_price) and not pd.isna(sma_50) and close_price > sma_50: buy_conditions.append(True) 
    if not pd.isna(bb_position) and bb_position < 0.2: buy_conditions.append(True) 
    if not (pd.isna(adx) or pd.isna(dmp) or pd.isna(dmn)) and adx > config_adx_threshold and dmp > dmn: buy_conditions.append(True) 

    # --- Sell Conditions ---
    if not pd.isna(rsi) and rsi > 70: sell_conditions.append(True) 
    if not pd.isna(ema_diff) and ema_diff < 0: sell_conditions.append(True) 
    if not (pd.isna(macd) or pd.isna(macd_signal)) and macd < macd_signal and macd_hist < 0: sell_conditions.append(True) 
    if not pd.isna(volume_ratio) and volume_ratio > 1.2: sell_conditions.append(True) 
    if not pd.isna(body) and body < 0: sell_conditions.append(True) 
    if not pd.isna(close_price) and not pd.isna(sma_50) and close_price < sma_50: sell_conditions.append(True) 
    if not pd.isna(bb_position) and bb_position > 0.8: sell_conditions.append(True) 
    if not (pd.isna(adx) or pd.isna(dmp) or pd.isna(dmn)) and adx > config_adx_threshold and dmn > dmp: sell_conditions.append(True) 
    
    # Calculate strength as a percentage of conditions met
    buy_strength = sum(buy_conditions) / len(buy_conditions) if buy_conditions else 0
    sell_strength = sum(sell_conditions) / len(sell_conditions) if sell_conditions else 0
    
    return buy_strength, sell_strength

def get_signal_strength_multi_timeframe(all_timeframe_data: dict, config_adx_threshold: int):
    """
    Aggregates signal strength across multiple timeframes.
    
    Args:
        all_timeframe_data (dict): Dictionary containing data for all monitored timeframes.
        config_adx_threshold (int): The ADX threshold from CONFIG.
        
    Returns:
        tuple: (aggregated_buy_strength: float, aggregated_sell_strength: float)
               Weighted scores between 0 and 1.
    """
    aggregated_buy_strength = 0
    aggregated_sell_strength = 0
    num_tfs_contributing = 0

    for tf_name, tf_data in all_timeframe_data.items():
        if tf_data.get('indicators_complete'): 
            latest_tf_candle_data = tf_data['latest_candle_data']
            buy_str, sell_str = get_signal_strength(latest_tf_candle_data, config_adx_threshold) 
            
            weight = 1.0
            if "M5" in tf_name: weight = 1.2
            elif "M15" in tf_name: weight = 1.5
            elif "H1" in tf_name: weight = 2.0
            elif "H4" in tf_name: weight = 2.5
            elif "D1" in tf_name: weight = 3.0

            aggregated_buy_strength += (buy_str * weight)
            aggregated_sell_strength += (sell_str * weight)
            num_tfs_contributing += weight 

    if num_tfs_contributing > 0:
        return aggregated_buy_strength / num_tfs_contributing, aggregated_sell_strength / num_tfs_contributing
    return 0, 0 

def get_multi_timeframe_confluence_status(all_timeframe_data: dict, config_mtf_confluence_threshold: float, config_adx_threshold: int): 
    """
    Determines the overall confluence status across multiple timeframes.
    Returns a string like "Strong Bullish", "Mixed", "Strong Bearish", "Conflicting".
    
    Args:
        all_timeframe_data (dict): Dictionary containing data for all monitored timeframes.
        config_mtf_confluence_threshold (float): Threshold for signal strength to count towards confluence.
        config_adx_threshold (int): The ADX threshold for calling get_signal_strength.
        
    Returns:
        Text: A Rich Text object indicating confluence status ("Strong Bullish", "Mixed", etc.).
    """
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    
    important_tfs = ["M5", "M15", "H1", "H4"] 
    
    for tf_name in important_tfs:
        tf_data = all_timeframe_data.get(tf_name, {})
        if tf_data.get('indicators_complete'):
            latest_tf_candle_data = tf_data['latest_candle_data']
            buy_str, sell_str = get_signal_strength(latest_tf_candle_data, config_adx_threshold) 
            
            if buy_str >= config_mtf_confluence_threshold:
                bullish_count += 1
            elif sell_str >= config_mtf_confluence_threshold:
                bearish_count += 1
            else:
                neutral_count += 1
    
    total_tfs_checked = bullish_count + bearish_count + neutral_count
    if total_tfs_checked == 0:
        return Text("N/A", style="dim") 

    if bullish_count > bearish_count and bullish_count > neutral_count:
        if bullish_count >= len(important_tfs) * 0.75: 
            return Text("Aligned Bullish", style="bold green")
        return Text("Bullish Bias", style="green")
    elif bearish_count > bullish_count and bearish_count > neutral_count:
        if bearish_count >= len(important_tfs) * 0.75: 
            return Text("Aligned Bearish", style="bold red")
        return Text("Bearish Bias", style="red")
    elif bullish_count > 0 and bearish_count > 0:
        return Text("Conflicting", style="orange") 
    else:
        return Text("Mixed/Neutral", style="white") 

def identify_high_probability_setup(data: dict, account_info, config_module, log_activity_func): 
    """
    Identifies a high-probability trade setup for a given symbol based on aggregated signals
    and risk management parameters.
    
    Args:
        data (dict): Dictionary containing live market data and indicator data.
        account_info (mt5.AccountInfo): MT5 account information.
        config_module (module): The loaded config module (e.g., `config.py` module).
        log_activity_func (function): Function to log activities.

    Returns:
        dict: A dictionary of setup details or None if no setup is identified.
    """
    symbol = data['symbol']
    current_price = data['current_price']
    bid = data['bid']
    ask = data['ask']
    symbol_info = data['symbol_info']

    # 1. Determine overall signal type based on aggregated multi-timeframe strength
    buy_strength, sell_strength = get_signal_strength_multi_timeframe(data['all_timeframe_data'], config_module.ADX_TREND_THRESHOLD) 

    signal_type = "NONE"
    if buy_strength >= config_module.HIGH_PROB_SIGNAL_THRESHOLD: 
        signal_type = "BUY"
    elif sell_strength >= config_module.HIGH_PROB_SIGNAL_THRESHOLD: 
        signal_type = "SELL"
    else:
        log_activity_func(f"Setup rejected for {symbol}: Aggregated signal strength ({buy_strength:.2f}B/{sell_strength:.2f}S) below threshold ({config_module.HIGH_PROB_SIGNAL_THRESHOLD}).", "dim") 
        return None

    # 2. Get data from a primary timeframe for calculating SL/TP and market conditions.
    primary_tf_for_setup = "H1" 
    if primary_tf_for_setup not in data['all_timeframe_data'] or not data['all_timeframe_data'][primary_tf_for_setup].get('indicators_complete'):
        primary_tf_for_setup = "M5" 
        if primary_tf_for_setup not in data['all_timeframe_data'] or not data['all_timeframe_data'][primary_tf_for_setup].get('indicators_complete'):
            primary_tf_for_setup = "M1" 
            if primary_tf_for_setup not in data['all_timeframe_data'] or not data['all_timeframe_data'][primary_tf_for_setup].get('indicators_complete'):
                log_activity_func(f"Setup rejected for {symbol}: No complete indicator data for any primary timeframe.", "dim")
                return None

    primary_tf_data_for_setup = data['all_timeframe_data'].get(primary_tf_for_setup, {})
    latest_candle_data = primary_tf_data_for_setup['latest_candle_data']
    
    # 3. Check for critical indicators for a robust setup (ADX and ATR)
    adx = _get_val(latest_candle_data, 'adx')
    atr = _get_val(latest_candle_data, 'atr')

    if pd.isna(adx) or adx < config_module.ADX_TREND_THRESHOLD: 
        log_activity_func(f"Setup rejected for {symbol}: ADX ({adx:.1f}) below trend threshold ({config_module.ADX_TREND_THRESHOLD}).", "dim") 
        return None
    if pd.isna(atr) or atr <= 0:
        log_activity_func(f"Setup rejected for {symbol}: Invalid ATR ({atr:.5f}).", "dim")
        return None

    # 4. Calculate Entry, Stop Loss, and Take Profit levels
    entry_price = ask if signal_type == "BUY" else bid 
    
    stop_loss = np.nan
    take_profit = np.nan
    
    if symbol_info is None: 
        log_activity_func(f"Setup rejected for {symbol}: Symbol info is None.", "red")
        return None 

    if not hasattr(symbol_info, 'point') or symbol_info.point == 0 or \
       not hasattr(symbol_info, 'trade_tick_value') or symbol_info.trade_tick_value == 0 or \
       not hasattr(symbol_info, 'trade_tick_size') or symbol_info.trade_tick_size == 0 or \
       not hasattr(symbol_info, 'trade_contract_size') or symbol_info.trade_contract_size == 0:
        log_activity_func(f"Setup rejected for {symbol}: Incomplete or zero symbol_info attributes (point, tick_value, tick_size, contract_size).", "red")
        return None 


    if signal_type == "BUY":
        stop_loss = entry_price - (atr * config_module.ATR_STOP_LOSS_MULTIPLIER) 
        if stop_loss >= entry_price: 
            log_activity_func(f"Setup rejected for {symbol}: Calculated BUY SL ({stop_loss:.{symbol_info.digits}f}) is not below entry ({entry_price:.{symbol_info.digits}f}).", "dim")
            return None 
        take_profit = entry_price + (abs(entry_price - stop_loss) * config_module.MIN_RISK_REWARD_RATIO) 
    elif signal_type == "SELL":
        stop_loss = entry_price + (atr * config_module.ATR_STOP_LOSS_MULTIPLIER) 
        if stop_loss <= entry_price: 
            log_activity_func(f"Setup rejected for {symbol}: Calculated SELL SL ({stop_loss:.{symbol_info.digits}f}) is not above entry ({entry_price:.{symbol_info.digits}f}).", "dim")
            return None 
        take_profit = entry_price - (abs(stop_loss - entry_price) * config_module.MIN_RISK_REWARD_RATIO) 

    # 5. Calculate Risk/Reward Ratio
    risk_reward_ratio = np.nan
    if not pd.isna(stop_loss) and not pd.isna(take_profit) and (entry_price - stop_loss) != 0:
        if signal_type == "BUY":
            if (entry_price - stop_loss) > 0:
                risk_reward_ratio = (take_profit - entry_price) / (entry_price - stop_loss)
        elif signal_type == "SELL":
            if (stop_loss - entry_price) > 0:
                risk_reward_ratio = (entry_price - take_profit) / (stop_loss - entry_price)

    if pd.isna(risk_reward_ratio) or risk_reward_ratio < config_module.MIN_RISK_REWARD_RATIO: 
        log_activity_func(f"Setup rejected for {symbol}: Risk/Reward ratio ({risk_reward_ratio:.2f}) below minimum ({config_module.MIN_RISK_REWARD_RATIO}).", "dim") 
        return None 

    # 6. Calculate Position Size (Lots) based on account equity and risk %
    risk_amount = account_info.equity * config_module.RISK_PER_TRADE_PERCENT 
    lots = np.nan

    if not pd.isna(stop_loss):
        value_per_point_per_lot = symbol_info.trade_contract_size * (symbol_info.trade_tick_value / symbol_info.trade_tick_size)
        
        if value_per_point_per_lot == 0:
            risk_per_lot_at_sl = np.nan
        else:
            risk_per_lot_at_sl = abs(entry_price - stop_loss) / symbol_info.point * value_per_point_per_lot

        if risk_per_lot_at_sl > 0:
            lots = risk_amount / risk_per_lot_at_sl
            
            lots = np.clip(lots, config_module.MIN_LOT_SIZE_CLI, config_module.MAX_LOT_SIZE_CLI) 
            lots = np.clip(lots, symbol_info.volume_min, symbol_info.volume_max)
            
            if symbol_info.volume_step > 0:
                lots = round(lots / symbol_info.volume_step) * symbol_info.volume_step
            
            if lots <= 0 and symbol_info.volume_min > 0:
                lots = symbol_info.volume_min 
                log_activity_func(f"Setup for {symbol}: Calculated lots adjusted up to min volume ({lots:.2f}).", "dim")
            elif lots <= 0:
                lots = 0.01 
                log_activity_func(f"Setup for {symbol}: Calculated lots adjusted to default 0.01.", "dim")

    if pd.isna(lots) or lots <= 0:
        log_activity_func(f"Setup rejected for {symbol}: Final lot size calculation resulted in invalid lots ({lots}).", "red")
        return None

    # 7. Calculate Potential Profit
    potential_profit = risk_amount * risk_reward_ratio if not pd.isna(risk_reward_ratio) else np.nan
    
    # 8. Extract Market Conditions from primary timeframe
    last_candle_type = "N/A"
    body = _get_val(latest_candle_data, 'body')
    if not pd.isna(body):
        if body > 0: last_candle_type = "BULLISH"
        elif body < 0: last_candle_type = "BEARISH"
        else: last_candle_type = "NEUTRAL (Doji)" 

    candle_size = _get_val(latest_candle_data, 'candle_size')
    volume_ratio = _get_val(latest_candle_data, 'volume_ratio')
    trend_strength = _get_val(latest_candle_data, 'adx')
    
    # 9. Calculate Setup Confidence (0-1) - combining various factors
    strength_component = (buy_strength if signal_type == "BUY" else sell_strength)
    trend_component = (adx / 100) if not pd.isna(adx) else 0 
    rr_component = min(1.0, (risk_reward_ratio / config_module.MIN_RISK_REWARD_RATIO) * 0.5) if not pd.isna(risk_reward_ratio) else 0 
    
    candle_confirm_component = 0
    if (signal_type == "BUY" and last_candle_type == "BULLISH") or \
       (signal_type == "SELL" and last_candle_type == "BEARISH"):
        candle_confirm_component = 0.1 

    setup_confidence = (strength_component * 0.5) + \
                       (trend_component * 0.25) + \
                       (rr_component * 0.15) + \
                       (candle_confirm_component * 0.1)
    
    setup_confidence = min(1.0, max(0.0, setup_confidence)) 

    log_activity_func(f"Identified potential {signal_type} setup for {symbol} with Confidence: {setup_confidence:.2f}.", "green")

    return {
        "Symbol": symbol,
        "Signal": signal_type,
        "Entry": entry_price,
        "Stop Loss": stop_loss,
        "Take Profit": take_profit,
        "Position Size": lots,
        "Risk/Reward": risk_reward_ratio,
        "Setup Confidence": setup_confidence, 
        "RSI": _get_val(latest_candle_data, 'rsi'),
        "Volume Ratio": volume_ratio,
        "Trend Strength": trend_strength, 
        "Risk Amount": risk_amount,
        "Potential Profit": potential_profit,
        "Last Candle": last_candle_type,
        "Candle Size": candle_size,
        "Volume vs Average": volume_ratio, 
        "Primary TF for Setup": primary_tf_for_setup,
        "symbol_info": symbol_info, 
        "aggregated_buy_strength": buy_strength, 
        "aggregated_sell_strength": sell_strength 
    }

