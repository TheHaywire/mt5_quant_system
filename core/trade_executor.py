import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging
from loguru import logger

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    "logs/trading_system_{time:YYYYMMDD}.log",
    rotation="00:00",  # New file at midnight
    retention="7 days",  # Keep logs for 7 days
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}"
)

class TradeExecutor:
    def __init__(self, config_module):
        self.config = config_module
        self.magic_number = config_module.MAGIC_NUMBER
        self.max_positions = config_module.MAX_OPEN_POSITIONS_PER_SYMBOL
        self.min_lot_size = 0.01
        self.max_lot_size = 1.0
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.min_rr_ratio = 1.5
        self.min_confidence = 0.6
        
        # Track active trades to avoid duplicates
        self.active_trades = set()
        
        # Initialize MT5 if not already initialized
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            raise RuntimeError("MT5 initialization failed")
            
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: float, account_equity: float) -> float:
        """Calculate position size based on risk management rules."""
        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Failed to get symbol info for {symbol}")
                return 0.0
                
            # Calculate risk amount in account currency
            risk_amount = account_equity * self.risk_per_trade
            
            # Calculate pip value
            point = symbol_info.point
            if symbol_info.digits == 3 or symbol_info.digits == 5:
                point *= 10
                
            # Calculate stop loss distance in points
            sl_distance = abs(entry_price - stop_loss) / point
            
            # Calculate position size
            tick_value = symbol_info.trade_tick_value
            if tick_value == 0:
                # For some symbols, we need to calculate tick value
                contract_size = symbol_info.trade_contract_size
                tick_value = (point * contract_size) / symbol_info.trade_tick_size
                
            position_size = risk_amount / (sl_distance * tick_value)
            
            # Round to allowed lot step
            lot_step = symbol_info.volume_step
            position_size = round(position_size / lot_step) * lot_step
            
            # Apply lot size limits
            position_size = max(self.min_lot_size, min(position_size, self.max_lot_size))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {str(e)}")
            return 0.0
            
    def validate_trade_setup(self, setup: Dict) -> bool:
        """Validate if a trade setup meets our criteria."""
        try:
            # Check basic requirements
            if not all(k in setup for k in ['Symbol', 'Signal', 'Entry', 'Stop Loss', 
                                          'Take Profit', 'Setup Confidence', 'Risk/Reward']):
                return False
                
            # Validate risk/reward ratio
            if setup['Risk/Reward'] < self.min_rr_ratio:
                logger.info(f"Trade rejected: R/R ratio {setup['Risk/Reward']:.2f} below minimum {self.min_rr_ratio}")
                return False
                
            # Validate confidence
            if setup['Setup Confidence'] < self.min_confidence:
                logger.info(f"Trade rejected: Confidence {setup['Setup Confidence']:.2f} below minimum {self.min_confidence}")
                return False
                
            # Check if we already have a position in this symbol
            symbol = setup['Symbol']
            positions = mt5.positions_get(symbol=symbol, magic=self.magic_number)
            if positions and len(positions) >= self.max_positions:
                logger.info(f"Trade rejected: Maximum positions ({self.max_positions}) reached for {symbol}")
                return False
                
            # Check if this is a duplicate setup
            setup_key = f"{symbol}_{setup['Signal']}_{setup['Entry']:.5f}"
            if setup_key in self.active_trades:
                logger.info(f"Trade rejected: Duplicate setup for {symbol}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade setup: {str(e)}")
            return False
            
    def execute_trade(self, setup: Dict, account_info) -> bool:
        """Execute a trade based on the validated setup."""
        try:
            symbol = setup['Symbol']
            
            # Validate the setup
            if not self.validate_trade_setup(setup):
                return False
                
            # Calculate position size
            position_size = self.calculate_position_size(
                symbol=symbol,
                entry_price=setup['Entry'],
                stop_loss=setup['Stop Loss'],
                account_equity=account_info.equity
            )
            
            if position_size < self.min_lot_size:
                logger.warning(f"Calculated position size {position_size} below minimum for {symbol}")
                return False
                
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position_size,
                "type": mt5.ORDER_TYPE_BUY if setup['Signal'] == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": setup['Entry'],
                "sl": setup['Stop Loss'],
                "tp": setup['Take Profit'],
                "deviation": 20,  # Maximum price deviation in points
                "magic": self.magic_number,
                "comment": f"QuantAI {setup['Primary TF for Setup']}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send the trade request
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Trade execution failed for {symbol}: {result.comment}")
                return False
                
            # Log successful trade
            logger.info(f"Trade executed for {symbol}: {setup['Signal']} {position_size} lots at {setup['Entry']}")
            
            # Add to active trades
            setup_key = f"{symbol}_{setup['Signal']}_{setup['Entry']:.5f}"
            self.active_trades.add(setup_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {setup['Symbol']}: {str(e)}")
            return False
            
    def cleanup_old_trades(self):
        """Remove old trade setups from active_trades set."""
        try:
            current_positions = mt5.positions_get(magic=self.magic_number)
            active_symbols = {pos.symbol for pos in current_positions} if current_positions else set()
            
            # Remove trades for symbols we no longer have positions in
            self.active_trades = {key for key in self.active_trades 
                                if key.split('_')[0] in active_symbols}
                                
        except Exception as e:
            logger.error(f"Error cleaning up old trades: {str(e)}")
            
    def process_trade_setups(self, trade_setups: list, account_info) -> None:
        """Process a list of trade setups and execute valid ones."""
        try:
            # Clean up old trades first
            self.cleanup_old_trades()
            
            # Sort setups by confidence (highest first)
            sorted_setups = sorted(trade_setups, 
                                 key=lambda x: x['Setup Confidence'], 
                                 reverse=True)
            
            # Process each setup
            for setup in sorted_setups:
                if self.execute_trade(setup, account_info):
                    # If trade executed successfully, wait a bit before next one
                    mt5.sleep(1000)  # 1 second delay between trades
                    
        except Exception as e:
            logger.error(f"Error processing trade setups: {str(e)}")
            
    def __del__(self):
        """Cleanup when the executor is destroyed."""
        mt5.shutdown() 