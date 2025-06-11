# If logging is needed, use:
import logging
# or
from loguru import logger 

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class MT5Connector:
    def __init__(self, config):
        self.config = config
        self.initialized = False
        self.connected = False
        self.last_connection_check = datetime.now()
        self.connection_check_interval = timedelta(minutes=5)
        
    def initialize(self) -> bool:
        """Initialize MT5 connection."""
        try:
            if not self.initialized:
                if not mt5.initialize(
                    path=self.config.MT5_PATH,
                    login=self.config.MT5_LOGIN,
                    password=self.config.MT5_PASSWORD,
                    server=self.config.MT5_SERVER
                ):
                    logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                    return False
                self.initialized = True
                logger.info("MT5 initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            return False
            
    def check_connection(self) -> bool:
        """Check and maintain MT5 connection."""
        try:
            current_time = datetime.now()
            if (current_time - self.last_connection_check) < self.connection_check_interval:
                return self.connected
                
            self.last_connection_check = current_time
            
            if not self.initialized:
                if not self.initialize():
                    return False
                    
            # Test connection by getting account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("MT5 connection lost")
                self.connected = False
                # Try to reinitialize
                self.initialized = False
                return self.initialize()
                
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error checking MT5 connection: {str(e)}")
            self.connected = False
            return False
            
    def get_account_info(self) -> Optional[mt5.AccountInfo]:
        """Get current account information."""
        try:
            if not self.check_connection():
                return None
            return mt5.account_info()
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None
            
    def get_symbol_info(self, symbol: str) -> Optional[mt5.SymbolInfo]:
        """Get symbol information."""
        try:
            if not self.check_connection():
                return None
            return mt5.symbol_info(symbol)
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            return None
            
    def get_ohlcv(self, symbol: str, timeframe: int, 
                  start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a symbol."""
        try:
            if not self.check_connection():
                return None
                
            rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
            if rates is None or len(rates) == 0:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {symbol}: {str(e)}")
            return None
            
    def get_positions(self, symbol: Optional[str] = None) -> List[mt5.TradePosition]:
        """Get current positions."""
        try:
            if not self.check_connection():
                return []
                
            if symbol:
                positions = mt5.positions_get(symbol=symbol, magic=self.config.MAGIC_NUMBER)
            else:
                positions = mt5.positions_get(magic=self.config.MAGIC_NUMBER)
                
            return positions if positions is not None else []
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
            
    def get_tick_data(self, symbol: str) -> Optional[Dict]:
        """Get current tick data for a symbol."""
        try:
            if not self.check_connection():
                return None
                
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"No tick data for {symbol}")
                return None
                
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time),
                'spread': tick.ask - tick.bid
            }
            
        except Exception as e:
            logger.error(f"Error getting tick data for {symbol}: {str(e)}")
            return None
            
    def __del__(self):
        """Cleanup when the connector is destroyed."""
        if self.initialized:
            mt5.shutdown() 