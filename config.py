# config.py

from collections import OrderedDict
import MetaTrader5 as mt5 # Needed for mt5.TIMEFRAME_ constants

# --- MT5 Connection Details ---
# IMPORTANT: Replace with your actual MT5 terminal path, login, password, and server.
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_LOGIN = 165835373
MT5_PASSWORD = "Manan@123!!"
MT5_SERVER = "XMGlobal-MT5 2"

# --- General Monitor Settings ---
NUM_CANDLES = 250 # Number of historical candles to fetch for indicator calculation
TABLE_REFRESH_SECONDS = 1.5 # How often the console output will print new data (effectively a delay)

# --- Indicator Thresholds & Parameters ---
# These influence signal strength and setup identification
ATR_STOP_LOSS_MULTIPLIER = 2.0 # Stop Loss distance = ATR * this multiplier
ADX_TREND_THRESHOLD = 25 # ADX value above which a trend is considered strong for setups
HIGH_PROB_SIGNAL_THRESHOLD = 0.7 # Minimum aggregated signal strength (0-1) for a high-probability setup
MTF_CONFLUENCE_THRESHOLD = 0.6 # Threshold for multi-timeframe signal agreement (0-1)

# --- Automated Trading Configuration ---
# IMPORTANT: Set to True for live trading, False for dry run/simulation.
# ALWAYS TEST ON A DEMO ACCOUNT FIRST!
LIVE_TRADING_ENABLED = False 

MAGIC_NUMBER = 20240610     # Unique ID for trades opened by THIS script (essential for managing your bot's trades)
MAX_OPEN_POSITIONS_PER_SYMBOL = 1 # Limit to one position per symbol for simplicity
MIN_LOT_SIZE_CLI = 0.1      # User-defined minimum lot size for automated trades
MAX_LOT_SIZE_CLI = 1.0      # User-defined maximum lot size for automated trades

# Strategy-specific thresholds for trade entry
# Setup Confidence must be >= this value to trigger an order
ENTRY_CONFIDENCE_THRESHOLD = 0.7 

# --- Risk Management Parameters ---
RISK_PER_TRADE_PERCENT = 0.01 # 1% risk per trade based on account equity
MIN_RISK_REWARD_RATIO = 1.5 # Minimum acceptable Risk/Reward for a high probability setup (e.g., 1:1.5)

# --- Position Management Parameters (for manage_open_position) ---
BREAK_EVEN_PROFIT_FACTOR = 1.0 # Move SL to breakeven when profit is X * initial risk
BREAK_EVEN_BUFFER_POINTS = 10 # Buffer for breakeven (e.g., 10 points above entry for BUY)
TRAILING_STOP_POINTS_INITIAL = 50 # Initial points of profit to activate trailing stop
TRAILING_STOP_POINTS_STEP = 20 # How many points the SL moves for each favorable movement (or ATR-based step)

# --- Symbols to Monitor (Categorized for better organization) ---
# Use OrderedDict to maintain insertion order for display
# Add more asset classes and tickers as needed, ensuring they are available on your MT5 broker.
# timeframes: Dictionary of mt5.TIMEFRAME_ constants to monitor for each symbol.
# name: A user-friendly name for display.
SYMBOLS_TO_MONITOR = OrderedDict([
    ("Forex Majors", OrderedDict([
        ("EURUSD", {"timeframes": {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}, "name": "EUR/USD"}),
        ("GBPUSD", {"timeframes": {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}, "name": "GBP/USD"}),
        ("USDJPY", {"timeframes": {"M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}, "name": "USD/JPY"}),
        ("AUDUSD", {"timeframes": {"M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1}, "name": "AUD/USD"}),
        ("USDCAD", {"timeframes": {"M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1}, "name": "USD/CAD"}),
        ("USDCHF", {"timeframes": {"M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1}, "name": "USD/CHF"}),
    ])),
    ("Forex Minors & Exotics", OrderedDict([
        ("EURJPY", {"timeframes": {"M15": mt5.TIMEFRAME_M15, "H4": mt5.TIMEFRAME_H4}, "name": "EUR/JPY"}),
        ("GBPJPY", {"timeframes": {"M15": mt5.TIMEFRAME_M15, "H4": mt5.TIMEFRAME_H4}, "name": "GBP/JPY"}),
        ("AUDCAD", {"timeframes": {"M15": mt5.TIMEFRAME_M15, "H4": mt5.TIMEFRAME_H4}, "name": "AUD/CAD"}),
        ("NZDUSD", {"timeframes": {"M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}, "name": "NZD/USD"}),
        ("USDMXN", {"timeframes": {"H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}, "name": "USD/MXN"}),
        ("EURZAR", {"timeframes": {"H1": mt5.TIMEFRAME_H1}, "name": "EUR/ZAR"}),
    ])),
    ("Metals", OrderedDict([
        ("GOLD", {"timeframes": {"M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}, "name": "GOLD"}),
        ("SILVER", {"timeframes": {"M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}, "name": "SILVER"}),
        ("XPTUSD", {"timeframes": {"H1": mt5.TIMEFRAME_H1}, "name": "PLATINUM"}), 
        ("XPDUSD", {"timeframes": {"H1": mt5.TIMEFRAME_H1}, "name": "PALLADIUM"}), 
    ])),
    ("Indices", OrderedDict([
        ("US30Cash", {"timeframes": {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}, "name": "US30 (Dow)"}),
        ("DE40Cash", {"timeframes": {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}, "name": "DAX 40"}),
        ("USTechCash", {"timeframes": {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}, "name": "US Tech 100"}),
        ("UK100Cash", {"timeframes": {"H1": mt5.TIMEFRAME_H1}, "name": "FTSE 100"}),
        ("JP225Cash", {"timeframes": {"H1": mt5.TIMEFRAME_H1}, "name": "Nikkei 225"}),
        ("AUS200Cash", {"timeframes": {"H1": mt5.TIMEFRAME_H1}, "name": "Australia 200"}),
    ])),
    ("Crypto", OrderedDict([
        ("BTCUSD", {"timeframes": {"M5": mt5.TIMEFRAME_M5, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}, "name": "Bitcoin"}),
        ("ETHUSD", {"timeframes": {"M5": mt5.TIMEFRAME_M5, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}, "name": "Ethereum"}),
        ("LTCUSD", {"timeframes": {"H1": mt5.TIMEFRAME_H1}, "name": "Litecoin"}),
        ("XRPUSD", {"timeframes": {"H1": mt5.TIMEFRAME_H1}, "name": "Ripple"}),
        ("ADAUSD", {"timeframes": {"H1": mt5.TIMEFRAME_H1}, "name": "Cardano"}),
        ("SOLUSD", {"timeframes": {"H1": mt5.TIMEFRAME_H1}, "name": "Solana"}),
    ])),
    ("Commodities", OrderedDict([
        ("USOil", {"timeframes": {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}, "name": "US Oil (WTI)"}),
        ("BRENTCash", {"timeframes": {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}, "name": "Brent Oil"}),
        ("NGAS", {"timeframes": {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}, "name": "Natural Gas"}),
    ])),
])
