import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import time
import numpy as np
import os
from collections import OrderedDict, deque # Import deque for activity log
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live # This import is now actively used
from rich import box
import logging
import sys
import traceback
from core.signal_validator import SignalValidator, SignalMetrics
import yaml

# Initialize Rich Console
console = Console()

# Define symbols to monitor with their timeframes
symbols_to_monitor = {
    "Forex": {
        "EURUSD": {"timeframes": ["M1", "M5", "H1", "H4", "D1"], "name": "EUR/USD"},
        "GBPUSD": {"timeframes": ["M1", "M5", "H1", "H4", "D1"], "name": "GBP/USD"},
        "USDJPY": {"timeframes": ["M1", "M5", "H1", "H4", "D1"], "name": "USD/JPY"},
        "AUDUSD": {"timeframes": ["M1", "M5", "H1", "H4", "D1"], "name": "AUD/USD"},
        "USDCAD": {"timeframes": ["M1", "M5", "H1", "H4", "D1"], "name": "USD/CAD"}
    },
    "Indices": {
        "US100Cash": {"timeframes": ["M5", "H1", "H4", "D1"], "name": "US Tech (Nasdaq)"},
        "US30Cash": {"timeframes": ["M5", "H1", "H4", "D1"], "name": "US30 (Dow)"},
        "UK100Cash": {"timeframes": ["M5", "H1", "H4", "D1"], "name": "UK100 (FTSE)"}
    },
    "Commodities": {
        "XAUUSD": {"timeframes": ["M5", "H1", "H4", "D1"], "name": "Gold"},
        "XAGUSD": {"timeframes": ["M5", "H1", "H4", "D1"], "name": "Silver"},
        "USOIL": {"timeframes": ["M5", "H1", "H4", "D1"], "name": "Crude Oil"}
    }
}

# --- Configuration Parameters ---
# These can be moved to a separate config.json or config.ini file for easier management
# and to avoid changing code for parameter tuning.
CONFIG = {
    "NUM_CANDLES": 250, # Number of historical candles to fetch for indicator calculation
    "TABLE_REFRESH_SECONDS": 1.5, # How often the console output table refreshes
    "RISK_PER_TRADE_PERCENT": 0.01, # 1% risk per trade based on account equity
    "MIN_RISK_REWARD_RATIO": 1.5, # Minimum acceptable Risk/Reward for a high probability setup (e.g., 1:1.5)
    "ATR_STOP_LOSS_MULTIPLIER": 2.0, # Stop Loss distance = ATR * this multiplier
    "ADX_TREND_THRESHOLD": 25, # ADX value above which a trend is considered strong for setups
    "HIGH_PROB_SIGNAL_THRESHOLD": 0.7, # Minimum aggregated signal strength (0-1) for a high-probability setup
    "MTF_CONFLUENCE_THRESHOLD": 0.6, # Threshold for multi-timeframe signal agreement (0-1)
    
    # --- NEW TRADING CONFIGURATION ---
    "LIVE_TRADING_ENABLED": True, # IMPORTANT: Set to TRUE for live trading, FALSE for dry run
    "MAGIC_NUMBER": 20240610,     # Unique ID for trades opened by THIS script
    "MAX_OPEN_POSITIONS_PER_SYMBOL": 1, # Limit to one position per symbol for simplicity
    "MIN_LOT_SIZE_CLI": 0.1,      # User-defined minimum lot size for automated trades
    "MAX_LOT_SIZE_CLI": 1.0,      # User-defined maximum lot size for automated trades
    "BREAK_EVEN_PROFIT_FACTOR": 1.0, # Move SL to breakeven when profit is X * initial risk
    "BREAK_EVEN_BUFFER_POINTS": 10, # Buffer for breakeven (e.g., 10 points above entry for BUY)
    "TRAILING_STOP_POINTS_INITIAL": 50, # Initial points for trailing stop activation
    "TRAILING_STOP_POINTS_STEP": 20, # How many points the SL moves for each favorable movement
    "ENTRY_CONFIDENCE_THRESHOLD": 0.65 # NEW: Minimum Setup Confidence to trigger a trade (lowered for testing)
}

# --- Global Activity Log (Managed by the rich.Live context) ---
# This deque will store recent activity messages for display in the footer
activity_log = deque(maxlen=10) # Keep last 10 messages

def log_activity(message: str, style: str = "white"):
    """Adds a timestamped message to the activity log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    activity_log.append(Text(f"[{timestamp}] {message}", style=style))

# --- Console Layout Setup ---
def make_layout() -> Layout:
    """Creates a Rich Layout for structured console output."""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3), # Top section for title and timestamp
        Layout(name="body", ratio=1),  # Main content area
        Layout(name="footer", size=5)  # Bottom section for messages/account info
    )
    # Split body into columns (vertical stacking)
    layout["body"].split_column(  # Changed from split_row
        Layout(name="market_snapshot", ratio=2), # Top panel in the body
        Layout(name="trade_setup", ratio=1)      # Bottom panel in the body
    )
    return layout

# --- Utility Functions ---
def clear_screen():
    """Clears the terminal screen.
    NOTE: When using rich.live.Live with screen=True, this might not be strictly necessary
    as Live handles screen clearing, but it's kept for robustness or
    if Live is ever disabled or in non-compatible terminals.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def connect_mt5():
    """Connects to MetaTrader 5 terminal with retry logic."""
    max_retries = 5
    # IMPORTANT: Replace with your actual MT5 terminal path if different
    mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe" 

    for attempt in range(max_retries):
        try:
            # Initialize MT5 connection
            if not mt5.initialize(path=mt5_path):
                log_activity(f"Failed to initialize MT5 (Attempt {attempt + 1}/{max_retries})", "red")
                time.sleep(3)
                continue
                
            # Log in to the MT5 account
            # IMPORTANT: Replace with your actual MT5 login details
            if not mt5.login(login=165835373, password="Manan@123!!", server="XMGlobal-MT5 2"):
                log_activity(f"Failed to login to MT5 (Attempt {attempt + 1}/{max_retries})", "red")
                mt5.shutdown() # Shutdown if login fails
                time.sleep(3)
                continue
                
            # Verify account information retrieval
            account_info = mt5.account_info()
            if account_info is None:
                log_activity(f"Failed to get account info (Attempt {attempt + 1}/{max_retries}). MT5 connection might be problematic.", "red")
                mt5.shutdown()
                time.sleep(3)
                continue
                
            log_activity("Connected successfully to MT5!", "green")
            return True
            
        except Exception as e:
            log_activity(f"Connection error: {str(e)}", "red")
            if attempt < max_retries - 1:
                log_activity("Retrying connection...", "yellow")
                time.sleep(3)
            else:
                return False # All retries failed
                
    return False # Should not be reached if max_retries attempts are made

# --- Indicator Calculation ---
def calculate_indicators(df):
    """
    Calculates various technical indicators for a given DataFrame.
    Returns a tuple (indicators_complete_status, latest_candle_data_dict).
    """
    if df.empty:
        return False, {}

    try:
        # Minimum candles needed for SMA(200) and ADX(14)
        min_candles_needed = max(200, 14) 

        # Ensure necessary columns for pandas_ta functions are present or initialized
        # This prevents errors if a column is missing in the initial DataFrame
        required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan # Add as NaN if missing

        if len(df) < min_candles_needed:
            # Not enough data to calculate all indicators, return partial data if any
            return False, df.iloc[-1].to_dict() if not df.empty else {}
        
        # Exponential Moving Averages (EMA)
        df['ema_fast'] = ta.ema(df['close'], length=8)
        df['ema_slow'] = ta.ema(df['close'], length=21)
        
        # Relative Strength Index (RSI)
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # Average True Range (ATR) - crucial for volatility-based SL/TP
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Moving Average Convergence Divergence (MACD)
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty and 'MACD_12_26_9' in macd_df.columns:
            df['macd'] = macd_df['MACD_12_26_9']
            df['macd_signal'] = macd_df['MACDs_12_26_9']
            df['macd_hist'] = macd_df['MACDh_12_26_9']
        else:
            # Fallback for MACD if pandas_ta has issues or column names vary
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volume Moving Average and Ratio
        if 'tick_volume' in df.columns and not df['tick_volume'].isnull().all():
            df['volume_ma'] = df['tick_volume'].rolling(20).mean()
            # Handle potential division by zero for volume_ma
            df['volume_ratio'] = df.apply(lambda row: row['tick_volume'] / row['volume_ma'] if row['volume_ma'] != 0 else np.nan, axis=1)
        else:
            df['volume_ma'] = np.nan
            df['volume_ratio'] = np.nan

        # Candle Anatomy
        df['body'] = df['close'] - df['open']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_size'] = abs(df['body'])
        df['candle_size'] = df['high'] - df['low'] # Total range of the candle
        
        # Simple Moving Averages (SMA)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        # Bollinger Bands (BBands)
        bbands_df = ta.bbands(df['close'], length=20)
        if bbands_df is not None and not bbands_df.empty and len(bbands_df.columns) >= 3:
            df['bb_upper'] = bbands_df.iloc[:, 0]
            df['bb_middle'] = bbands_df.iloc[:, 1]
            df['bb_lower'] = bbands_df.iloc[:, 2]
        else:
            df['bb_upper'] = np.nan
            df['bb_middle'] = np.nan
            df['bb_lower'] = np.nan

        # Average Directional Index (ADX) - for trend strength
        adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_data is not None and not adx_data.empty and 'ADX_14' in adx_data.columns:
            df['adx'] = adx_data['ADX_14']
            df['dmp'] = adx_data['DMP_14'] # +DI
            df['dmn'] = adx_data['DMN_14'] # -DI
        else:
            df['adx'] = np.nan
            df['dmp'] = np.nan
            df['dmn'] = np.nan
        
        # Return True for complete indicators and the last row as a dictionary
        return True, df.iloc[-1].to_dict()
        
    except Exception as e:
        log_activity(f"Error calculating indicators: {str(e)}", "red")
        # Return False if an error occurs, but still try to return last candle data if possible
        return False, df.iloc[-1].to_dict() if not df.empty else {}

# --- Data Retrieval from MT5 ---
def get_mt5_symbol_info(symbol: str) -> dict:
    """Get detailed information about a symbol from MT5."""
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        
        return {
            "name": info.name,
            "description": info.description,
            "point": info.point,
            "digits": info.digits,
            "spread": info.spread,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_mode": info.trade_mode,
            "visible": info.visible,
            "session_deals": info.session_deals,
            "session_buy_orders": info.session_buy_orders,
            "session_sell_orders": info.session_sell_orders,
            "volume": info.volume,
            "time": info.time,
            "last": info.last,
            "bid": info.bid,
            "ask": info.ask
        }
    except Exception as e:
        log_activity(f"Error getting symbol info for {symbol}: {str(e)}", "red")
        return None

def validate_and_map_symbols():
    """Validate and map symbols to their correct MT5 names."""
    # Common symbol mappings for different brokers
    symbol_mappings = {
        # Indices
        "USTechCash": ["USTEC.cash", "USTECm", "USTEC", "USTEC.cash", "USTECm.cash"],
        "UK100": ["UK100.cash", "UK100m", "UK100", "UK100.cash", "UK100m.cash"],
        # Commodities
        "USOil": ["XTIUSD", "USOIL", "WTIUSD", "USOIL.cash", "USOILm"],
        "NATGAS": ["XNGUSD", "NATGAS", "NATGAS.cash", "NATGASm", "NGAS"]
    }
    
    validated_symbols = {}
    failed_symbols = []

    # First, get all available symbols from MT5
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        log_activity("Failed to get symbols from MT5", "red")
        return None
    
    available_symbols = [s.name for s in all_symbols]
    log_activity(f"Available symbols in MT5: {', '.join(available_symbols)}", "green")

    # Validate each symbol in our monitoring list
    for category, symbols_dict in symbols_to_monitor.items():
        for symbol, symbol_info in symbols_dict.items():
            # Check if symbol exists as is
            if symbol in available_symbols:
                validated_symbols[symbol] = symbol
                continue

            # Try alternative names from mapping
            if symbol in symbol_mappings:
                for alt_name in symbol_mappings[symbol]:
                    if alt_name in available_symbols:
                        validated_symbols[symbol] = alt_name
                        log_activity(f"Symbol mapped: {symbol} -> {alt_name}", "green")
                        break
                else:
                    failed_symbols.append(symbol)
                    log_activity(f"Failed to find valid symbol for {symbol}. Tried: {', '.join(symbol_mappings[symbol])}", "red")
            else:
                failed_symbols.append(symbol)
                log_activity(f"No mapping found for symbol: {symbol}", "red")

    # Log validation results
    if failed_symbols:
        log_activity(f"Failed to validate symbols: {', '.join(failed_symbols)}", "red")
    
    # Update symbols_to_monitor with validated symbols
    for category in symbols_to_monitor:
        for symbol in list(symbols_to_monitor[category].keys()):
            if symbol in validated_symbols:
                # Update the symbol name if it was mapped
                if validated_symbols[symbol] != symbol:
                    symbol_info = symbols_to_monitor[category][symbol]
                    del symbols_to_monitor[category][symbol]
                    symbols_to_monitor[category][validated_symbols[symbol]] = symbol_info
            else:
                # Remove invalid symbols
                del symbols_to_monitor[category][symbol]

    return validated_symbols

def get_live_data(symbol, timeframes_to_fetch, num_candles=CONFIG["NUM_CANDLES"]):
    """Get live market data with improved error handling and validation."""
    try:
        # First check if symbol exists and is enabled
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            log_activity(f"Symbol {symbol} not found in MT5", "red")
            return None

        # Try to enable the symbol if it's not visible
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                log_activity(f"Failed to enable symbol {symbol}", "red")
                return None
            log_activity(f"Enabled symbol {symbol}", "green")

        # Get detailed symbol info
        symbol_details = get_mt5_symbol_info(symbol)
        if symbol_details is None:
            log_activity(f"Failed to get details for {symbol}", "red")
            return None

        # Log symbol details for debugging
        log_activity(f"Symbol {symbol} details: Point={symbol_details['point']}, "
                    f"Digits={symbol_details['digits']}, Spread={symbol_details['spread']}, "
                    f"Volume={symbol_details['volume']}", "dim")

        # Rest of the existing get_live_data function...
        # ... (keep existing code) ...

    except Exception as e:
        log_activity(f"Error in get_live_data for {symbol}: {str(e)}", "red")
        return None

# --- Signal Strength Calculation ---
def get_signal_strength(latest_data_dict):
    """
    Calculates buy/sell signal strength based on the latest indicators
    from a single timeframe's candle data.
    """
    buy_conditions = []
    sell_conditions = []

    # Helper to safely retrieve indicator values, defaulting to NaN if not present
    def get_val(key, default=np.nan):
        val = latest_data_dict.get(key, default)
        return val if not pd.isna(val) else np.nan

    # Retrieve indicator values
    rsi = get_val('rsi')
    ema_fast = get_val('ema_fast')
    ema_slow = get_val('ema_slow')
    macd = get_val('macd')
    macd_signal = get_val('macd_signal')
    macd_hist = get_val('macd_hist')
    volume_ratio = get_val('volume_ratio')
    close_price = get_val('close')
    open_price = get_val('open')
    sma_50 = get_val('sma_50')
    sma_200 = get_val('sma_200')
    bb_upper = get_val('bb_upper')
    bb_lower = get_val('bb_lower')
    adx = get_val('adx')
    dmp = get_val('dmp') # +DI
    dmn = get_val('dmn') # -DI

    # Calculate differences/ratios
    ema_diff = ema_fast - ema_slow if not (pd.isna(ema_fast) or pd.isna(ema_slow)) else np.nan
    body = close_price - open_price if not (pd.isna(close_price) or pd.isna(open_price)) else np.nan

    bb_position = np.nan
    if not (pd.isna(close_price) or pd.isna(bb_upper) or pd.isna(bb_lower)) and (bb_upper - bb_lower) != 0:
        bb_position = (close_price - bb_lower) / (bb_upper - bb_lower)

    # --- Buy Conditions (True if condition met, False otherwise for strength calculation) ---
    if not pd.isna(rsi) and rsi < 30: buy_conditions.append(True) # Oversold condition
    if not pd.isna(ema_diff) and ema_diff > 0: buy_conditions.append(True) # EMA bullish crossover
    if not (pd.isna(macd) or pd.isna(macd_signal)) and macd > macd_signal and macd_hist > 0: buy_conditions.append(True) # MACD bullish crossover with positive histogram
    if not pd.isna(volume_ratio) and volume_ratio > 1.2: buy_conditions.append(True) # Higher than average volume
    if not pd.isna(body) and body > 0: buy_conditions.append(True) # Bullish candle close > open
    if not pd.isna(close_price) and not pd.isna(sma_50) and close_price > sma_50: buy_conditions.append(True) # Price above 50 SMA
    if not pd.isna(bb_position) and bb_position < 0.2: buy_conditions.append(True) # Price near lower Bollinger Band
    if not (pd.isna(adx) or pd.isna(dmp) or pd.isna(dmn)) and adx > CONFIG["ADX_TREND_THRESHOLD"] and dmp > dmn: buy_conditions.append(True) # Strong uptrend confirmation with ADX

    # --- Sell Conditions ---
    if not pd.isna(rsi) and rsi > 70: sell_conditions.append(True) # Overbought condition
    if not pd.isna(ema_diff) and ema_diff < 0: sell_conditions.append(True) # EMA bearish crossover
    if not (pd.isna(macd) or pd.isna(macd_signal)) and macd < macd_signal and macd_hist < 0: sell_conditions.append(True) # MACD bearish crossover with negative histogram
    if not pd.isna(volume_ratio) and volume_ratio > 1.2: sell_conditions.append(True) # Higher than average volume
    if not pd.isna(body) and body < 0: sell_conditions.append(True) # Bearish candle close < open
    if not pd.isna(close_price) and not pd.isna(sma_50) and close_price < sma_50: sell_conditions.append(True) # Price below 50 SMA
    if not pd.isna(bb_position) and bb_position > 0.8: sell_conditions.append(True) # Price near upper Bollinger Band
    if not (pd.isna(adx) or pd.isna(dmp) or pd.isna(dmn)) and adx > CONFIG["ADX_TREND_THRESHOLD"] and dmn > dmp: sell_conditions.append(True) # Strong downtrend confirmation with ADX
    
    # Calculate strength as a percentage of conditions met
    buy_strength = sum(buy_conditions) / len(buy_conditions) if buy_conditions else 0
    sell_strength = sum(sell_conditions) / len(sell_conditions) if sell_conditions else 0
    
    return buy_strength, sell_strength

def get_signal_strength_multi_timeframe(all_timeframe_data):
    """
    Aggregates signal strength across multiple timeframes.
    This can be customized with weighted averages or stricter confluence rules.
    """
    aggregated_buy_strength = 0
    aggregated_sell_strength = 0
    num_tfs_contributing = 0

    # Iterate through each timeframe's data
    for tf_name, tf_data in all_timeframe_data.items():
        if tf_data.get('indicators_complete'): # Only use timeframes with complete indicator data
            latest_tf_candle_data = tf_data['latest_candle_data']
            buy_str, sell_str = get_signal_strength(latest_tf_candle_data)
            
            # Assign weights to different timeframes (e.g., higher TFs have more weight)
            weight = 1.0
            if "M5" in tf_name: weight = 1.2
            elif "M15" in tf_name: weight = 1.5
            elif "H1" in tf_name: weight = 2.0
            elif "H4" in tf_name: weight = 2.5
            elif "D1" in tf_name: weight = 3.0

            aggregated_buy_strength += (buy_str * weight)
            aggregated_sell_strength += (sell_str * weight)
            num_tfs_contributing += weight # Sum of weights for averaging

    # Calculate the average aggregated strength
    if num_tfs_contributing > 0:
        return aggregated_buy_strength / num_tfs_contributing, aggregated_sell_strength / num_tfs_contributing
    return 0, 0 # Return zero if no timeframes contributed data

def get_multi_timeframe_confluence_status(all_timeframe_data):
    """
    Determines the overall confluence status across multiple timeframes.
    Returns a string like "Strong Bullish", "Mixed", "Strong Bearish", "Conflicting".
    """
    bullish_count = 0
    bearish_count = 0
    neutral_count = 0
    
    # Define a set of important timeframes to check for confluence
    important_tfs = ["M5", "M15", "H1", "H4"] 
    
    for tf_name in important_tfs:
        tf_data = all_timeframe_data.get(tf_name, {})
        if tf_data.get('indicators_complete'):
            latest_tf_candle_data = tf_data['latest_candle_data']
            buy_str, sell_str = get_signal_strength(latest_tf_candle_data)
            
            if buy_str >= CONFIG["MTF_CONFLUENCE_THRESHOLD"]:
                bullish_count += 1
            elif sell_str >= CONFIG["MTF_CONFLUENCE_THRESHOLD"]:
                bearish_count += 1
            else:
                neutral_count += 1
    
    total_tfs_checked = bullish_count + bearish_count + neutral_count
    if total_tfs_checked == 0:
        return Text("N/A", style="dim") # No relevant timeframe data

    # Simple majority rule for confluence
    if bullish_count > bearish_count and bullish_count > neutral_count:
        if bullish_count >= len(important_tfs) * 0.75: # Strong alignment
            return Text("Aligned Bullish", style="bold green")
        return Text("Bullish Bias", style="green")
    elif bearish_count > bullish_count and bearish_count > neutral_count:
        if bearish_count >= len(important_tfs) * 0.75: # Strong alignment
            return Text("Aligned Bearish", style="bold red")
        return Text("Bearish Bias", style="red")
    elif bullish_count > 0 and bearish_count > 0:
        return Text("Conflicting", style="orange") # Strong opposing signals
    else:
        return Text("Mixed/Neutral", style="white") # No clear bias or not enough strong signals

# --- High Probability Setup Identification ---
def identify_high_probability_setup(data, account_info):
    """
    Identifies a high-probability trade setup for a given symbol based on aggregated signals
    and risk management parameters.
    Returns a dictionary of setup details or None if no setup is identified.
    """
    symbol = data['symbol']
    current_price = data['current_price']
    bid = data['bid']
    ask = data['ask']
    symbol_info = data['symbol_info']

    # 1. Determine overall signal type based on aggregated multi-timeframe strength
    buy_strength, sell_strength = get_signal_strength_multi_timeframe(data['all_timeframe_data'])

    signal_type = "NONE"
    if buy_strength >= CONFIG["HIGH_PROB_SIGNAL_THRESHOLD"]:
        signal_type = "BUY"
    elif sell_strength >= CONFIG["HIGH_PROB_SIGNAL_THRESHOLD"]:
        signal_type = "SELL"
    else:
        log_activity(f"Setup rejected for {symbol}: Aggregated signal strength ({buy_strength:.2f}B/{sell_strength:.2f}S) below threshold ({CONFIG['HIGH_PROB_SIGNAL_THRESHOLD']}).", "dim")
        return None # No strong signal for a high probability setup

    # 2. Get data from a primary timeframe for calculating SL/TP and market conditions.
    # Prioritize higher timeframes for stability if available, or fall back.
    # The primary timeframe used here will be passed to the setup details for clarity.
    primary_tf_for_setup = "H1" 
    if primary_tf_for_setup not in data['all_timeframe_data'] or not data['all_timeframe_data'][primary_tf_for_setup].get('indicators_complete'):
        primary_tf_for_setup = "M5" 
        if primary_tf_for_setup not in data['all_timeframe_data'] or not data['all_timeframe_data'][primary_tf_for_setup].get('indicators_complete'):
            primary_tf_for_setup = "M1" 
            if primary_tf_for_setup not in data['all_timeframe_data'] or not data['all_timeframe_data'][primary_tf_for_setup].get('indicators_complete'):
                log_activity(f"Setup rejected for {symbol}: No complete indicator data for any primary timeframe.", "dim")
                return None # Cannot proceed without any primary timeframe data

    primary_tf_data_for_setup = data['all_timeframe_data'].get(primary_tf_for_setup, {})
    latest_candle_data = primary_tf_data_for_setup['latest_candle_data']
    
    # 3. Check for critical indicators for a robust setup (ADX and ATR)
    adx = latest_candle_data.get('adx')
    atr = latest_candle_data.get('atr')

    if pd.isna(adx) or adx < CONFIG["ADX_TREND_THRESHOLD"]:
        log_activity(f"Setup rejected for {symbol}: ADX ({adx:.1f}) below trend threshold ({CONFIG['ADX_TREND_THRESHOLD']}).", "dim")
        return None # Not a strong enough trend for a high probability setup
    if pd.isna(atr) or atr <= 0:
        log_activity(f"Setup rejected for {symbol}: Invalid ATR ({atr:.5f}).", "dim")
        return None # ATR not available or invalid (critical for dynamic SL/TP)

    # 4. Calculate Entry, Stop Loss, and Take Profit levels
    # Entry price can be current bid/ask. Using bid for sell, ask for buy.
    entry_price = ask if signal_type == "BUY" else bid 
    
    stop_loss = np.nan
    take_profit = np.nan
    
    # Ensure symbol_info is available for accurate calculations (pip/lot value)
    if symbol_info is None: 
        log_activity(f"Setup rejected for {symbol}: Symbol info is None.", "red")
        return None 

    # Handle cases where crucial symbol info attributes might be missing or zero
    if not hasattr(symbol_info, 'point') or symbol_info.point == 0 or \
       not hasattr(symbol_info, 'trade_tick_value') or symbol_info.trade_tick_value == 0 or \
       not hasattr(symbol_info, 'trade_tick_size') or symbol_info.trade_tick_size == 0 or \
       not hasattr(symbol_info, 'trade_contract_size') or symbol_info.trade_contract_size == 0:
        log_activity(f"Setup rejected for {symbol}: Incomplete or zero symbol_info attributes (point, tick_value, tick_size, contract_size).", "red")
        return None # Cannot calculate reliable setup without full symbol info


    if signal_type == "BUY":
        stop_loss = entry_price - (atr * CONFIG["ATR_STOP_LOSS_MULTIPLIER"])
        # Ensure TP is above entry, and SL is below entry for BUY
        if stop_loss >= entry_price: 
            log_activity(f"Setup rejected for {symbol}: Calculated BUY SL ({stop_loss:.{symbol_info.digits}f}) is not below entry ({entry_price:.{symbol_info.digits}f}).", "dim")
            return None 
        take_profit = entry_price + (abs(entry_price - stop_loss) * CONFIG["MIN_RISK_REWARD_RATIO"])
    elif signal_type == "SELL":
        stop_loss = entry_price + (atr * CONFIG["ATR_STOP_LOSS_MULTIPLIER"])
        # Ensure TP is below entry, and SL is above entry for SELL
        if stop_loss <= entry_price: 
            log_activity(f"Setup rejected for {symbol}: Calculated SELL SL ({stop_loss:.{symbol_info.digits}f}) is not above entry ({entry_price:.{symbol_info.digits}f}).", "dim")
            return None 
        take_profit = entry_price - (abs(stop_loss - entry_price) * CONFIG["MIN_RISK_REWARD_RATIO"])

    # 5. Calculate Risk/Reward Ratio
    risk_reward_ratio = np.nan
    if not pd.isna(stop_loss) and not pd.isna(take_profit) and (entry_price - stop_loss) != 0:
        if signal_type == "BUY":
            if (entry_price - stop_loss) > 0:
                risk_reward_ratio = (take_profit - entry_price) / (entry_price - stop_loss)
        elif signal_type == "SELL":
            if (stop_loss - entry_price) > 0:
                risk_reward_ratio = (entry_price - take_profit) / (stop_loss - entry_price)

    # Only consider setups that meet the minimum R:R
    if pd.isna(risk_reward_ratio) or risk_reward_ratio < CONFIG["MIN_RISK_REWARD_RATIO"]:
        log_activity(f"Setup rejected for {symbol}: Risk/Reward ratio ({risk_reward_ratio:.2f}) below minimum ({CONFIG['MIN_RISK_REWARD_RATIO']}).", "dim")
        return None 

    # 6. Calculate Position Size (Lots) based on account equity and risk %
    risk_amount = account_info.equity * CONFIG["RISK_PER_TRADE_PERCENT"]
    lots = np.nan

    if not pd.isna(stop_loss):
        # Calculate risk per lot in account currency
        value_per_point_per_lot = symbol_info.trade_contract_size * (symbol_info.trade_tick_value / symbol_info.trade_tick_size)
        
        if value_per_point_per_lot == 0:
            risk_per_lot_at_sl = np.nan
        else:
            risk_per_lot_at_sl = abs(entry_price - stop_loss) / symbol_info.point * value_per_point_per_lot

        if risk_per_lot_at_sl > 0:
            lots = risk_amount / risk_per_lot_at_sl
            
            # --- NEW: Clamp lots to user-defined MIN/MAX and broker's limits ---
            lots = np.clip(lots, CONFIG["MIN_LOT_SIZE_CLI"], CONFIG["MAX_LOT_SIZE_CLI"])
            lots = np.clip(lots, symbol_info.volume_min, symbol_info.volume_max)
            
            # Round lots to the nearest step size
            if symbol_info.volume_step > 0:
                lots = round(lots / symbol_info.volume_step) * symbol_info.volume_step
            
            # Final check to ensure lots is not zero if it's supposed to be traded
            if lots <= 0 and symbol_info.volume_min > 0:
                lots = symbol_info.volume_min 
                log_activity(f"Setup for {symbol}: Calculated lots adjusted up to min volume ({lots:.2f}).", "dim")
            elif lots <= 0:
                lots = 0.01 
                log_activity(f"Setup for {symbol}: Calculated lots adjusted to default 0.01.", "dim")

    if pd.isna(lots) or lots <= 0:
        log_activity(f"Setup rejected for {symbol}: Final lot size calculation resulted in invalid lots ({lots}).", "red")
        return None

    # 7. Calculate Potential Profit
    potential_profit = risk_amount * risk_reward_ratio if not pd.isna(risk_reward_ratio) else np.nan
    
    # 8. Extract Market Conditions from primary timeframe
    last_candle_type = "N/A"
    body = latest_candle_data.get('body')
    if not pd.isna(body):
        if body > 0: last_candle_type = "BULLISH"
        elif body < 0: last_candle_type = "BEARISH"
        else: last_candle_type = "NEUTRAL (Doji)" 

    candle_size = latest_candle_data.get('candle_size', np.nan)
    volume_ratio = latest_candle_data.get('volume_ratio', np.nan)
    trend_strength = latest_candle_data.get('adx', np.nan) # ADX value
    
    # 9. Calculate Setup Confidence (0-1) - combining various factors
    # This score is a more comprehensive measure of setup quality.
    
    # Signal strength component (from aggregated MTF)
    strength_component = (buy_strength if signal_type == "BUY" else sell_strength)
    
    # Trend component (normalize ADX)
    trend_component = (adx / 100) if not pd.isna(adx) else 0 # Normalize ADX to 0-1 range
    
    # Risk/Reward component (normalize R:R) - give bonus for higher R:R
    rr_component = min(1.0, (risk_reward_ratio / CONFIG["MIN_RISK_REWARD_RATIO"]) * 0.5) if not pd.isna(risk_reward_ratio) else 0

    # Candle confirmation component (simple bonus for aligned candle)
    candle_confirm_component = 0
    if (signal_type == "BUY" and last_candle_type == "BULLISH") or \
       (signal_type == "SELL" and last_candle_type == "BEARISH"):
        candle_confirm_component = 0.1 # Small bonus

    # Weighted average for overall confidence
    setup_confidence = (strength_component * 0.5) + \
                       (trend_component * 0.25) + \
                       (rr_component * 0.15) + \
                       (candle_confirm_component * 0.1)
    
    setup_confidence = min(1.0, max(0.0, setup_confidence)) # Clamp between 0 and 1

    log_activity(f"Identified potential {signal_type} setup for {symbol} with Confidence: {setup_confidence:.2f}.", "green")

    return {
        "Symbol": symbol,
        "Signal": signal_type,
        "Entry": entry_price,
        "Stop Loss": stop_loss,
        "Take Profit": take_profit,
        "Position Size": lots,
        "Risk/Reward": risk_reward_ratio,
        "Setup Confidence": setup_confidence, # Renamed from Signal Score
        "RSI": latest_candle_data.get('rsi'),
        "Volume Ratio": volume_ratio,
        "Trend Strength": trend_strength, # ADX value
        "Risk Amount": risk_amount,
        "Potential Profit": potential_profit,
        "Last Candle": last_candle_type,
        "Candle Size": candle_size,
        "Volume vs Average": volume_ratio, # Redundant but matches prompt's requirement
        "Primary TF for Setup": primary_tf_for_setup, # Indicate which TF was used for calculations
        "symbol_info": symbol_info # Pass symbol_info with the setup for formatting
    }

# --- Trading Functions ---
def execute_order(symbol: str, order_type: str, lots: float, entry_price: float, sl: float, tp: float, comment: str = "") -> bool:
    """
    Execute a market order with proper error handling and validation.
    Returns True if order was placed successfully, False otherwise.
    """
    try:
        # Validate inputs
        if not symbol or not order_type or lots <= 0 or entry_price <= 0:
            log_activity(f"Invalid order parameters for {symbol}", "red")
            return False

        # Prepare the order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,  # Maximum price deviation in points
            "magic": 234000,  # Unique identifier for our orders
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send the order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log_activity(f"Order failed for {symbol}: {result.comment}", "red")
            return False

        log_activity(f"Order executed for {symbol}: {order_type} {lots} lots at {entry_price}", "green")
        return True

    except Exception as e:
        log_activity(f"Error executing order for {symbol}: {str(e)}", "red")
        return False

def modify_position(position, new_sl: float = None, new_tp: float = None) -> bool:
    """Modify an existing position's stop loss or take profit."""
    try:
        if position is None:
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": position.ticket,
            "sl": new_sl if new_sl is not None else position.sl,
            "tp": new_tp if new_tp is not None else position.tp
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log_activity(f"Failed to modify position {position.ticket}: {result.comment}", "red")
            return False

        log_activity(f"Modified position {position.ticket} for {position.symbol}", "green")
        return True

    except Exception as e:
        log_activity(f"Error modifying position: {str(e)}", "red")
        return False

def get_open_positions(symbol: str = None) -> list:
    """Get all open positions, optionally filtered by symbol."""
    try:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        return positions if positions is not None else []
    except Exception as e:
        log_activity(f"Error getting positions: {str(e)}", "red")
        return []

def manage_open_position(position, current_price, symbol_info, latest_candle_data):
    """
    Manages an open position (e.g., move SL to breakeven, trailing stop).
    """
    # Get current ATR from the primary timeframe for the symbol
    # The primary TF for position management should ideally be retrieved from the position's comment/ticket if stored
    # For now, we assume it's passed or try to default.
    # In a real system, you'd store the primary TF for the setup when you opened the trade
    # e.g., in the position comment or a separate database.
    
    # Try to extract the primary TF from the position comment (if stored in send_trade_order)
    primary_tf_for_pos_mgmt = "H1" # Default fallback
    if position and hasattr(position, 'comment') and '_TF:' in position.comment:
        try:
            primary_tf_for_pos_mgmt = position.comment.split('_TF:')[-1].strip()
        except Exception:
            pass # Use default

    atr = latest_candle_data.get('atr')
    if pd.isna(atr) or atr <= 0:
        # log_activity(f"Warning: ATR not available for {position.symbol} for position management. Cannot apply BE/TS.", "yellow")
        return # Cannot manage without ATR or if data is incomplete

    modified = False
    new_sl = position.sl # Current SL
    new_tp = position.tp # Current TP
    
    # Get price precision for rounding SL/TP
    price_point = symbol_info.point
    
    # 1. Break-Even Logic
    # Calculate profit in points
    if position.type == mt5.POSITION_TYPE_BUY:
        profit_points = (current_price - position.price_open) / price_point
        breakeven_sl = position.price_open + (CONFIG["BREAK_EVEN_BUFFER_POINTS"] * price_point)
        # Only move to BE if profit is at least X times the initial risk (SL distance)
        # And if the current SL is still below the breakeven point
        initial_risk_points = (position.price_open - position.sl) / price_point if position.sl > 0 else 0
        
        if initial_risk_points > 0 and profit_points >= initial_risk_points * CONFIG["BREAK_EVEN_PROFIT_FACTOR"]:
            # Round breakeven_sl to the nearest tick
            breakeven_sl = round(breakeven_sl / price_point) * price_point
            if new_sl < breakeven_sl: # Only move if new SL is better than current
                new_sl = breakeven_sl
                modified = True
                log_activity(f"Position {position.ticket} ({position.symbol}): Moving SL to breakeven ({new_sl:.{symbol_info.digits}f}).", "green")

    elif position.type == mt5.POSITION_TYPE_SELL:
        profit_points = (position.price_open - current_price) / price_point
        breakeven_sl = position.price_open - (CONFIG["BREAK_EVEN_BUFFER_POINTS"] * price_point)
        initial_risk_points = (position.sl - position.price_open) / price_point if position.sl > 0 else 0

        if initial_risk_points > 0 and profit_points >= initial_risk_points * CONFIG["BREAK_EVEN_PROFIT_FACTOR"]:
            # Round breakeven_sl to the nearest tick
            breakeven_sl = round(breakeven_sl / price_point) * price_point
            if new_sl > breakeven_sl or new_sl == 0: # Only move if new SL is better/valid
                new_sl = breakeven_sl
                modified = True
                log_activity(f"Position {position.ticket} ({position.symbol}): Moving SL to breakeven ({new_sl:.{symbol_info.digits}f}).", "green")

    # 2. Trailing Stop Logic (Simple ATR-based trailing stop)
    # Only trail if profit is already substantial (e.g., beyond initial trailing stop activation points)
    # And if SL has been moved to breakeven or beyond.
    if position.type == mt5.POSITION_TYPE_BUY:
        current_profit_points = (current_price - position.price_open) / price_point
        
        # If profit exceeds initial trailing stop activation points and SL can be moved higher
        if current_profit_points >= CONFIG["TRAILING_STOP_POINTS_INITIAL"] and \
           (new_sl < (current_price - (atr * price_point)) or new_sl <= position.price_open): # Check against entry price as well
            
            proposed_sl = current_price - (atr * price_point * 0.5) # Trail SL at half ATR distance from current price (can be adjusted)
            proposed_sl = round(proposed_sl / price_point) * price_point # Round to tick size

            # Only move SL if it means locking in more profit (or if current SL is 0)
            if proposed_sl > new_sl:
                new_sl = proposed_sl
                modified = True
                log_activity(f"Position {position.ticket} ({position.symbol}): Trailing SL to {new_sl:.{symbol_info.digits}f}.", "green")

    elif position.type == mt5.POSITION_TYPE_SELL:
        current_profit_points = (position.price_open - current_price) / price_point
        
        if current_profit_points >= CONFIG["TRAILING_STOP_POINTS_INITIAL"] and \
           (new_sl > (current_price + (atr * price_point)) or new_sl >= position.price_open): # Check against entry price
            
            proposed_sl = current_price + (atr * price_point * 0.5)
            proposed_sl = round(proposed_sl / price_point) * price_point

            if proposed_sl < new_sl or new_sl == 0:
                new_sl = proposed_sl
                modified = True
                log_activity(f"Position {position.ticket} ({position.symbol}): Trailing SL to {new_sl:.{symbol_info.digits}f}.", "green")

    # Execute modification if needed and SL/TP values have actually changed
    if modified and (position.sl != new_sl or position.tp != new_tp):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": new_sl,
            "tp": new_tp,
            "magic": CONFIG["MAGIC_NUMBER"],
            "comment": position.comment, # Preserve original comment
        }
        if CONFIG["LIVE_TRADING_ENABLED"]:
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                log_activity(f"Error modifying position {position.ticket}: {result.retcode} - {result.comment}", "red")
            else:
                log_activity(f"Modified position {position.ticket} ({position.symbol}) to SL: {new_sl:.{symbol_info.digits}f}, TP: {new_tp:.{symbol_info.digits}f}.", "green")
        else:
            log_activity(f"DRY RUN: Would modify position {position.ticket}. New SL: {new_sl:.{symbol_info.digits}f}, New TP: {new_tp:.{symbol_info.digits}f}", "yellow")

# --- Data Formatting for Rich Table ---
def format_data_for_table(data):
    """
    Formats relevant data for a table row, including styling for Rich Table.
    Uses the M1 timeframe data for the main overview table by default.
    """
    formatted_row = {}

    # Helper to safely get value and format or return "N/A"
    def get_val_fmt(data_dict, key, fmt_str=None, default='N/A'):
        val = data_dict.get(key, np.nan)
        if pd.isna(val) or val is None:
            return default
        if fmt_str:
            return fmt_str.format(val)
        return str(val)

    # Try to get M1 data for the main table. If not available, use the first available timeframe.
    # This prioritizes the lowest timeframe for real-time snapshot.
    primary_tf_for_display = "M1"
    m1_data = data.get('all_timeframe_data', {}).get(primary_tf_for_display, {})

    # If M1 data is not complete or not available, try other timeframes in order
    if not m1_data.get('indicators_complete', False):
        # Ordered list of timeframes from most granular to broadest for fallback
        for tf_name in ["M5", "M15", "H1", "H4", "D1"]: 
            if data.get('all_timeframe_data', {}).get(tf_name, {}).get('indicators_complete', False):
                m1_data = data['all_timeframe_data'][tf_name]
                primary_tf_for_display = tf_name # Update which TF was used for display
                break

    indicators_complete_for_display_tf = m1_data.get('indicators_complete', False)
    latest = m1_data.get('latest_candle_data', {})


    if not indicators_complete_for_display_tf:
        # Return default values for incomplete data with "N/A" and dim style
        formatted_row = {
            'Symbol': Text(data['symbol'] if data else 'N/A', style="dim"),
            'Price': 'N/A', 'Spread': 'N/A', 'RSI': 'N/A', 'MACD Hist': 'N/A',
            'EMA Trend': 'N/A', 'MA Trend': 'N/A', 'Vol. Ratio': 'N/A', 'BB Pos': 'N/A',
            'Buy Str': 'N/A', 'Sell Str': 'N/A', 'Signal': Text("NO DATA", style="dim yellow"),
            'Sentiment': Text("N/A", style="dim"),
            'MTF Confluence': Text("N/A", style="dim") # New column
        }
    else:
        # Use aggregated signal strength from multiple timeframes for overall signal/sentiment
        buy_strength, sell_strength = get_signal_strength_multi_timeframe(data['all_timeframe_data'])

        signal_text = "NEUTRAL"
        signal_style = "white"
        sentiment_text = "Neutral"
        sentiment_style = "white"

        # Determine overall signal and sentiment based on aggregated strength
        if buy_strength >= 0.7:
            signal_text = "STRONG BUY"
            signal_style = "bold green"
            sentiment_text = "Bullish"
            sentiment_style = "green"
        elif sell_strength >= 0.7:
            signal_text = "STRONG SELL"
            signal_style = "bold red"
            sentiment_text = "Bearish"
            sentiment_style = "red"
        elif buy_strength >= 0.5:
            signal_text = "BUY"
            signal_style = "green"
            sentiment_text = "Slightly Bullish"
            sentiment_style = "green"
        elif sell_strength >= 0.5:
            signal_text = "SELL"
            signal_style = "red"
            sentiment_text = "Slightly Bearish"
            sentiment_style = "red"
        
        # EMA Trend (from the chosen display timeframe's data)
        ema_trend_str = "N/A"
        ema_trend_style = "white"
        ema_fast = latest.get('ema_fast')
        ema_slow = latest.get('ema_slow')
        if not (pd.isna(ema_fast) or pd.isna(ema_slow)):
            ema_diff = ema_fast - ema_slow
            if ema_diff > 0: 
                ema_trend_str = "BULL"
                ema_trend_style = "green"
            elif ema_diff < 0: 
                ema_trend_str = "BEAR"
                ema_trend_style = "red"
            else: ema_trend_str = "FLAT"

        # MA Trend (50 vs 200 from the chosen display timeframe's data)
        ma_trend_str = "N/A"
        ma_trend_style = "white"
        sma_50 = latest.get('sma_50')
        sma_200 = latest.get('sma_200')
        if not (pd.isna(sma_50) or pd.isna(sma_200)):
            if sma_50 > sma_200: 
                ma_trend_str = "BULL"
                ma_trend_style = "green"
            elif sma_50 < sma_200: 
                ma_trend_str = "BEAR"
                ma_trend_style = "red"
            else: ma_trend_str = "FLAT"
        
        # BB Position (from the chosen display timeframe's data)
        bb_position_str = "N/A"
        close_price = latest.get('close')
        bb_upper = latest.get('bb_upper')
        bb_lower = latest.get('bb_lower')

        if not (pd.isna(close_price) or pd.isna(bb_upper) or pd.isna(bb_lower)) and (bb_upper - bb_lower) != 0:
            bb_pos = (close_price - bb_lower) / (bb_upper - bb_lower)
            bb_position_str = f"{bb_pos:.0%}"

        # Determine price formatting based on symbol digits for accuracy (e.g., 5 for EURUSD, 2 for XAUUSD)
        price_fmt_str = "{:.5f}" # Default to 5 decimal places
        if data.get('symbol_info') and hasattr(data['symbol_info'], 'digits'):
            price_fmt_str = f"{{:.{data['symbol_info'].digits}f}}"

        # Get Multi-Timeframe Confluence status
        mtf_confluence_status = get_multi_timeframe_confluence_status(data['all_timeframe_data'])

        # Construct the formatted row for the Rich Table
        formatted_row = {
            'Symbol': Text(data['symbol'], style="bold cyan"),
            'Price': get_val_fmt(data, 'current_price', price_fmt_str),
            'Spread': get_val_fmt(data, 'spread', "{:.1f}", default='N/A'), 
            'RSI': get_val_fmt(latest, 'rsi', "{:.1f}"),
            'MACD Hist': get_val_fmt(latest, 'macd_hist', "{:.5f}"),
            'EMA Trend': Text(ema_trend_str, style=ema_trend_style),
            'MA Trend': Text(ma_trend_str, style=ma_trend_style),
            'Vol. Ratio': get_val_fmt(latest, 'volume_ratio', "{:.2f}"),
            'BB Pos': bb_position_str,
            'Buy Str': get_val_fmt({'strength': buy_strength}, 'strength', "{:.0%}"),
            'Sell Str': get_val_fmt({'strength': sell_strength}, 'strength', "{:.0%}"),
            'Signal': Text(signal_text, style=signal_style),
            'Sentiment': Text(sentiment_text, style=sentiment_style),
            'MTF Confluence': mtf_confluence_status # New column
        }
    
    return formatted_row

def validate_mt5_symbols():
    """Validate which symbols are actually available in the MT5 terminal."""
    if not mt5.initialize():
        log_activity("Failed to initialize MT5", "red")
        return None
        
    # Get all available symbols
    symbols = mt5.symbols_get()
    if symbols is None:
        log_activity("Failed to get symbols from MT5", "red")
        return None
        
    # Create a list of available symbols
    available_symbols = [s.name for s in symbols]
    log_activity(f"Available symbols in MT5: {', '.join(available_symbols)}", "green")
    return available_symbols

# Load signal validation config
with open('config/signal_config.yaml', 'r') as f:
    SIGNAL_CONFIG = yaml.safe_load(f)

# Initialize signal validator
signal_validator = SignalValidator(lookback_periods=SIGNAL_CONFIG['lookback_periods'])

def check_for_trade_setup(data):
    """
    Identifies a high-probability trade setup using quantitative validation.
    Returns a dictionary of setup details or None if no setup is identified.
    """
    symbol = data['symbol']
    current_price = data['current_price']
    bid = data['bid']
    ask = data['ask']
    symbol_info = data['symbol_info']

    # 1. Determine overall signal type based on aggregated multi-timeframe strength
    buy_strength, sell_strength = get_signal_strength_multi_timeframe(data['all_timeframe_data'])

    signal_type = "NONE"
    if buy_strength >= CONFIG["HIGH_PROB_SIGNAL_THRESHOLD"]:
        signal_type = "BUY"
    elif sell_strength >= CONFIG["HIGH_PROB_SIGNAL_THRESHOLD"]:
        signal_type = "SELL"
    else:
        log_activity(f"Setup rejected for {symbol}: Aggregated signal strength ({buy_strength:.2f}B/{sell_strength:.2f}S) below threshold ({CONFIG['HIGH_PROB_SIGNAL_THRESHOLD']}).", "dim")
        return None

    # 2. Get primary timeframe data for validation
    primary_tf_for_setup = "H1"  # Default to H1
    if primary_tf_for_setup not in data['all_timeframe_data'] or not data['all_timeframe_data'][primary_tf_for_setup].get('indicators_complete'):
        # Try other timeframes in order of preference
        for tf in ["M5", "M15", "H4", "D1"]:
            if tf in data['all_timeframe_data'] and data['all_timeframe_data'][tf].get('indicators_complete'):
                primary_tf_for_setup = tf
                break
        else:
            log_activity(f"Setup rejected for {symbol}: No complete indicator data for any primary timeframe.", "dim")
            return None

    # 3. Prepare indicators for validation
    primary_tf_data = data['all_timeframe_data'][primary_tf_for_setup]
    latest_candle_data = primary_tf_data['latest_candle_data']
    
    # Convert indicators to pandas Series for validation
    indicators = {
        'adx': pd.Series([latest_candle_data.get('adx', np.nan)]),
        'ema_fast': pd.Series([latest_candle_data.get('ema_fast', np.nan)]),
        'ema_slow': pd.Series([latest_candle_data.get('ema_slow', np.nan)]),
        'rsi': pd.Series([latest_candle_data.get('rsi', np.nan)]),
        'macd': pd.Series([latest_candle_data.get('macd', np.nan)]),
        'macd_signal': pd.Series([latest_candle_data.get('macd_signal', np.nan)]),
        'macd_hist': pd.Series([latest_candle_data.get('macd_hist', np.nan)])
    }

    # 4. Validate signal using quantitative metrics
    # Get symbol-specific thresholds if available
    min_confidence = SIGNAL_CONFIG['symbol_overrides'].get(symbol, {}).get('min_confidence', 
                    SIGNAL_CONFIG['timeframes'].get(primary_tf_for_setup, {}).get('min_confidence',
                    SIGNAL_CONFIG['min_confidence']))

    is_valid, metrics = signal_validator.validate_signal(
        signal_type=signal_type,
        symbol=symbol,
        timeframe=primary_tf_for_setup,
        indicators=indicators,
        min_confidence=min_confidence
    )

    if not is_valid:
        log_activity(f"Setup rejected for {symbol}: Failed quantitative validation. "
                    f"Confidence: {metrics.confidence:.2f}, Z-score: {metrics.z_score:.2f}, "
                    f"Trend: {metrics.trend_strength:.2f}", "dim")
        return None

    # 5. Calculate entry, stop loss, and take profit levels
    atr = latest_candle_data.get('atr')
    if pd.isna(atr) or atr <= 0:
        log_activity(f"Setup rejected for {symbol}: Invalid ATR ({atr:.5f}).", "dim")
        return None

    entry_price = ask if signal_type == "BUY" else bid
    
    # Calculate SL/TP using ATR
    if signal_type == "BUY":
        stop_loss = entry_price - (atr * CONFIG["ATR_STOP_LOSS_MULTIPLIER"])
        if stop_loss >= entry_price:
            log_activity(f"Setup rejected for {symbol}: Invalid BUY SL ({stop_loss:.{symbol_info.digits}f}).", "dim")
            return None
        take_profit = entry_price + (abs(entry_price - stop_loss) * CONFIG["MIN_RISK_REWARD_RATIO"])
    else:  # SELL
        stop_loss = entry_price + (atr * CONFIG["ATR_STOP_LOSS_MULTIPLIER"])
        if stop_loss <= entry_price:
            log_activity(f"Setup rejected for {symbol}: Invalid SELL SL ({stop_loss:.{symbol_info.digits}f}).", "dim")
            return None
        take_profit = entry_price - (abs(stop_loss - entry_price) * CONFIG["MIN_RISK_REWARD_RATIO"])

    # 6. Calculate position size using risk management
    risk_amount = mt5.account_info().equity * CONFIG["RISK_PER_TRADE_PERCENT"]
    value_per_point_per_lot = symbol_info.trade_contract_size * (symbol_info.trade_tick_value / symbol_info.trade_tick_size)
    
    if value_per_point_per_lot == 0:
        log_activity(f"Setup rejected for {symbol}: Invalid point value calculation.", "red")
        return None

    risk_per_lot_at_sl = abs(entry_price - stop_loss) / symbol_info.point * value_per_point_per_lot
    lots = risk_amount / risk_per_lot_at_sl if risk_per_lot_at_sl > 0 else 0

    # Apply lot size constraints
    lots = np.clip(lots, CONFIG["MIN_LOT_SIZE_CLI"], CONFIG["MAX_LOT_SIZE_CLI"])
    lots = np.clip(lots, symbol_info.volume_min, symbol_info.volume_max)
    if symbol_info.volume_step > 0:
        lots = round(lots / symbol_info.volume_step) * symbol_info.volume_step

    if lots <= 0:
        log_activity(f"Setup rejected for {symbol}: Invalid lot size ({lots:.2f}).", "red")
        return None

    # 7. Calculate risk/reward ratio
    risk_reward_ratio = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
    if risk_reward_ratio < CONFIG["MIN_RISK_REWARD_RATIO"]:
        log_activity(f"Setup rejected for {symbol}: Risk/Reward ratio ({risk_reward_ratio:.2f}) below minimum ({CONFIG['MIN_RISK_REWARD_RATIO']}).", "dim")
        return None

    # 8. Prepare setup details with quantitative metrics
    setup = {
        "Symbol": symbol,
        "Signal": signal_type,
        "Entry": entry_price,
        "Stop Loss": stop_loss,
        "Take Profit": take_profit,
        "Position Size": lots,
        "Risk/Reward": risk_reward_ratio,
        "Setup Confidence": metrics.confidence,
        "Z-Score": metrics.z_score,
        "Regime Score": metrics.regime_score,
        "Trend Strength": metrics.trend_strength,
        "Volatility Score": metrics.volatility_score,
        "Signal Quality": metrics.signal_quality,
        "Risk Amount": risk_amount,
        "Potential Profit": risk_amount * risk_reward_ratio,
        "Primary TF": primary_tf_for_setup,
        "symbol_info": symbol_info
    }

    log_activity(f"Valid setup identified for {symbol} {signal_type}: "
                f"Confidence={metrics.confidence:.2f}, "
                f"Z-score={metrics.z_score:.2f}, "
                f"Trend={metrics.trend_strength:.2f}", "green")

    return setup

# --- Main Execution Loop ---
def main():
    """Main function to run the trading system."""
    # Initialize MT5 first
    if not mt5.initialize():
        log_activity("Failed to initialize MT5", "red")
        return

    # Then connect with credentials
    if not connect_mt5():
        log_activity("Failed to connect to MT5 account", "red")
        mt5.shutdown()
        return

    # Get all available symbols
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        log_activity("Failed to get symbols from MT5", "red")
        mt5.shutdown()
        return

    # Log available symbols for debugging
    available_symbols = [s.name for s in all_symbols]
    log_activity(f"Available symbols in MT5: {', '.join(available_symbols)}", "green")

    # Update symbols_to_monitor with only available symbols
    for category in list(symbols_to_monitor.keys()):
        for symbol in list(symbols_to_monitor[category].keys()):
            # Try different variations of the symbol name
            symbol_variations = [
                symbol,
                f"{symbol}.cash",
                f"{symbol}m",
                f"{symbol}m.cash"
            ]
            
            # Special cases for specific symbols
            if symbol == "USTechCash":
                symbol_variations.extend(["USTEC", "USTEC.cash", "USTECm"])
            elif symbol == "UK100":
                symbol_variations.extend(["UK100.cash", "UK100m"])
            elif symbol == "USOil":
                symbol_variations.extend(["XTIUSD", "USOIL", "WTIUSD"])
            elif symbol == "NATGAS":
                symbol_variations.extend(["XNGUSD", "NGAS"])

            # Try to find a valid symbol name
            found = False
            for var in symbol_variations:
                if var in available_symbols:
                    if var != symbol:
                        # Update the symbol name if it was mapped
                        symbol_info = symbols_to_monitor[category][symbol]
                        del symbols_to_monitor[category][symbol]
                        symbols_to_monitor[category][var] = symbol_info
                        log_activity(f"Symbol mapped: {symbol} -> {var}", "green")
                    found = True
                    break

            if not found:
                log_activity(f"Symbol not found in MT5: {symbol}. Tried variations: {', '.join(symbol_variations)}", "red")
                del symbols_to_monitor[category][symbol]

    # Remove empty categories
    for category in list(symbols_to_monitor.keys()):
        if not symbols_to_monitor[category]:
            del symbols_to_monitor[category]

    # Log final symbol list
    log_activity("Final symbol list:", "green")
    for category, symbols in symbols_to_monitor.items():
        log_activity(f"{category}: {', '.join(symbols.keys())}", "green")

    # Rest of the main function...
    try:
        while True:
            # Process each symbol
            for category, symbols_dict in symbols_to_monitor.items():
                for symbol, symbol_info in symbols_dict.items():
                    try:
                        # Get market data
                        data = get_live_data(symbol, symbol_info["timeframes"])
                        if data is None:
                            continue

                        # Check for trade setup
                        setup = check_for_trade_setup(data)
                        if setup is not None:
                            # Execute trade if setup is valid
                            if setup["Setup Confidence"] >= CONFIG["ENTRY_CONFIDENCE_THRESHOLD"]:
                                success = execute_order(
                                    symbol=symbol,
                                    order_type=setup["Signal Type"],
                                    lots=setup["Position Size"],
                                    entry_price=setup["Entry Price"],
                                    sl=setup["Stop Loss"],
                                    tp=setup["Take Profit"],
                                    comment=f"Auto {setup['Signal Type']} {setup['Setup Confidence']:.2f}"
                                )
                                if success:
                                    log_activity(f"Trade executed for {symbol}: {setup['Signal Type']}", "green")

                        # Check and manage existing positions
                        positions = get_open_positions(symbol)
                        for position in positions:
                            # Add your position management logic here
                            pass

                    except Exception as e:
                        log_activity(f"Error processing {symbol}: {str(e)}", "red")
                        continue

            # Sleep to prevent excessive CPU usage
            time.sleep(CONFIG["TABLE_REFRESH_SECONDS"])

    except KeyboardInterrupt:
        log_activity("Shutting down...", "yellow")
    finally:
        mt5.shutdown()


# --- Script Entry Point ---
if __name__ == "__main__":
    main()

