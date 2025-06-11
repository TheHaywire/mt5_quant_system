# main.py

import MetaTrader5 as mt5
import time
import gc
import psutil
import logging
from datetime import datetime, timedelta
from collections import OrderedDict 
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn 
import traceback
import sys
import os

# Import modules
import config 
import mt5_utils 
import indicators 
import strategy 
import trade_manager 
import display_manager 
from core.trade_executor import TradeExecutor

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'trading_system_{datetime.now().strftime("%Y%m%d")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Access the console instance and activity log from display_manager
console = display_manager.console
activity_log = display_manager.activity_log
log_activity = display_manager.log_activity

def check_memory_usage():
    """Monitor memory usage and trigger garbage collection if needed."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    if memory_percent > 80:  # If using more than 80% of available memory
        logging.warning(f"High memory usage detected: {memory_percent:.1f}%")
        gc.collect()  # Force garbage collection
        return True
    return False

def check_mt5_connection():
    """Verify MT5 connection and attempt reconnection if needed."""
    try:
        if not mt5.initialize():
            logging.error("MT5 initialization failed")
            return False
            
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info")
            mt5.shutdown()
            return False
            
        return True
    except Exception as e:
        logging.error(f"MT5 connection check failed: {str(e)}")
        return False

def reconnect_mt5():
    """Attempt to reconnect to MT5 with retries."""
    max_retries = 5
    retry_delay = 30  # seconds
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempting MT5 reconnection (attempt {attempt + 1}/{max_retries})")
            mt5.shutdown()  # Ensure clean shutdown
            time.sleep(5)  # Wait before reconnecting
            
            if mt5_utils.connect_mt5(config.MT5_PATH, config.MT5_LOGIN, config.MT5_PASSWORD, config.MT5_SERVER, log_activity):
                logging.info("MT5 reconnection successful")
                return True
                
            logging.warning(f"Reconnection attempt {attempt + 1} failed")
            time.sleep(retry_delay)
            
        except Exception as e:
            logging.error(f"Reconnection error: {str(e)}")
            time.sleep(retry_delay)
            
    logging.error("All reconnection attempts failed")
    return False

def main():
    """
    Main function to connect to MT5, fetch data, calculate indicators,
    identify trade setups, and manage trades in real-time.
    """
    last_connection_check = datetime.now()
    connection_check_interval = timedelta(minutes=5)
    last_memory_check = datetime.now()
    memory_check_interval = timedelta(minutes=15)
    
    # Initialize trade executor
    trade_executor = TradeExecutor(config)
    
    try:
        while True:  # Outer loop for continuous operation
            try:
                # --- MT5 Connection ---
                if not mt5_utils.connect_mt5(config.MT5_PATH, config.MT5_LOGIN, config.MT5_PASSWORD, config.MT5_SERVER, log_activity):
                    logging.error("Initial MT5 connection failed")
                    if not reconnect_mt5():
                        time.sleep(300)  # Wait 5 minutes before retrying
                        continue
                    
                account_info = mt5.account_info()
                if account_info is None:
                    logging.error("Failed to get account info")
                    mt5.shutdown()
                    time.sleep(300)
                    continue
                
                # Print initial account information
                logging.info(f"Connected to account: {account_info.name} (Login: {account_info.login})")
                logging.info(f"Balance: ${account_info.balance:.2f}, Equity: ${account_info.equity:.2f}")
                
                # Calculate total symbols to monitor
                total_symbols_to_monitor_count = sum(len(tickers) for tickers in config.SYMBOLS_TO_MONITOR.values())
                
                # --- Initial Data Fetch and Processing ---
                initial_all_symbols_data_for_display = []
                initial_trade_setups = []
                
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    TimeElapsedColumn(),
                    console=console 
                ) as progress:
                    data_fetch_task = progress.add_task(
                        f"[cyan]Initializing data for {total_symbols_to_monitor_count} symbols...", 
                        total=total_symbols_to_monitor_count
                    )
                    
                    symbol_processed_count = 0
                    for category_name, tickers_in_category in config.SYMBOLS_TO_MONITOR.items():
                        for symbol_mt5_name, symbol_config in tickers_in_category.items():
                            data = mt5_utils.get_live_data(
                                symbol_mt5_name, 
                                symbol_config["timeframes"], 
                                config.NUM_CANDLES, 
                                log_activity, 
                                indicators.calculate_indicators
                            )
                            symbol_processed_count += 1
                            progress.update(data_fetch_task, advance=1, description=f"[cyan]Processing {symbol_mt5_name} ({symbol_processed_count}/{total_symbols_to_monitor_count})...[/cyan]")

                            if data:
                                aggregated_buy_strength, aggregated_sell_strength = strategy.get_signal_strength_multi_timeframe(
                                    data['all_timeframe_data'], config.ADX_TREND_THRESHOLD
                                )
                                data['aggregated_buy_strength'] = aggregated_buy_strength
                                data['aggregated_sell_strength'] = aggregated_sell_strength

                                data['mtf_confluence_text'] = strategy.get_multi_timeframe_confluence_status(
                                    data['all_timeframe_data'], config.MTF_CONFLUENCE_THRESHOLD, config.ADX_TREND_THRESHOLD
                                )

                                initial_all_symbols_data_for_display.append(data) # Store the full data dict, not just formatted rows
                                
                                setup = strategy.identify_high_probability_setup(data, account_info, config, log_activity)
                                if setup:
                                    initial_trade_setups.append(setup) 

                                # Automated Trading Logic (Entry only during initial fetch)
                                if setup and not mt5.positions_get(symbol=symbol_mt5_name, magic=config.MAGIC_NUMBER): 
                                    if setup['Setup Confidence'] >= config.ENTRY_CONFIDENCE_THRESHOLD: 
                                        trade_type = mt5.ORDER_TYPE_BUY if setup['Signal'] == "BUY" else mt5.ORDER_TYPE_SELL
                                        trade_manager.send_trade_order(
                                            setup, trade_type, config.MAGIC_NUMBER, config.LIVE_TRADING_ENABLED, log_activity
                                        )
                            else:
                                # Append placeholder data that can be formatted for display
                                initial_all_symbols_data_for_display.append({
                                    'symbol': symbol_config["name"], 
                                    'current_price': 0, 'spread': 0, 'all_timeframe_data': {}, 
                                    'symbol_info': None, 'aggregated_buy_strength': 0, 
                                    'aggregated_sell_strength': 0, 
                                    'mtf_confluence_text': display_manager.Text("NO DATA", style="dim yellow")
                                })

                log_activity("Initial data fetch complete. Starting continuous monitoring.", "bold green")
                
                # Use the initial data for the first continuous display refresh
                # We will re-format these into display lines within the loop
                all_symbols_data_dicts = initial_all_symbols_data_for_display
                trade_setups = initial_trade_setups

                # --- Continuous Monitoring Loop ---
                last_table_refresh_time = time.time()
                
                while True:  # Inner loop for normal operation
                    try:
                        current_time = datetime.now()
                        
                        # Periodic connection check
                        if current_time - last_connection_check >= connection_check_interval:
                            if not check_mt5_connection():
                                logging.warning("MT5 connection check failed, attempting reconnection")
                                if not reconnect_mt5():
                                    raise Exception("MT5 reconnection failed")
                            last_connection_check = current_time
                        
                        # Periodic memory check
                        if current_time - last_memory_check >= memory_check_interval:
                            if check_memory_usage():
                                logging.info("Memory cleanup performed")
                            last_memory_check = current_time
                        
                        # --- Only fetch new data and update internal lists if refresh time elapsed ---
                        if time.time() - last_table_refresh_time >= config.TABLE_REFRESH_SECONDS:
                            log_activity(f"Refreshing data for display cycle.", "dim") 
                            
                            # These lists will hold the data for the new cycle
                            current_cycle_all_symbols_data_dicts = [] 
                            current_cycle_trade_setups = [] 
                            
                            # Fetch and Process Data for Each Symbol across Categories (Continuous)
                            for category_name, tickers_in_category in config.SYMBOLS_TO_MONITOR.items():
                                for symbol_mt5_name, symbol_config in tickers_in_category.items():
                                    data = mt5_utils.get_live_data(
                                        symbol_mt5_name, 
                                        symbol_config["timeframes"], 
                                        config.NUM_CANDLES, 
                                        log_activity, 
                                        indicators.calculate_indicators
                                    )

                                    if data:
                                        aggregated_buy_strength, aggregated_sell_strength = strategy.get_signal_strength_multi_timeframe(
                                            data['all_timeframe_data'], config.ADX_TREND_THRESHOLD
                                        )
                                        data['aggregated_buy_strength'] = aggregated_buy_strength
                                        data['aggregated_sell_strength'] = aggregated_sell_strength
                                        data['mtf_confluence_text'] = strategy.get_multi_timeframe_confluence_status(
                                            data['all_timeframe_data'], config.MTF_CONFLUENCE_THRESHOLD, config.ADX_TREND_THRESHOLD
                                        )
                                        current_cycle_all_symbols_data_dicts.append(data) # Store full data dict
                                        
                                        setup = strategy.identify_high_probability_setup(data, account_info, config, log_activity)
                                        if setup:
                                            current_cycle_trade_setups.append(setup) 

                                        # --- Automated Trading Logic (Continuous Loop) ---
                                        all_mt5_positions = mt5.positions_get(symbol=symbol_mt5_name)
                                        bot_open_positions_for_symbol = [
                                            pos for pos in all_mt5_positions
                                            if pos.magic == config.MAGIC_NUMBER
                                        ]

                                        # 1. Entry Logic
                                        if setup and not bot_open_positions_for_symbol:
                                            if setup['Setup Confidence'] >= config.ENTRY_CONFIDENCE_THRESHOLD: 
                                                trade_type = mt5.ORDER_TYPE_BUY if setup['Signal'] == "BUY" else mt5.ORDER_TYPE_SELL
                                                trade_manager.send_trade_order(
                                                    setup, trade_type, config.MAGIC_NUMBER, config.LIVE_TRADING_ENABLED, log_activity
                                                )
                                        # 2. Position Management Logic
                                        if bot_open_positions_for_symbol:
                                            for position in bot_open_positions_for_symbol:
                                                position_primary_tf = position.comment.split('_TF:')[-1].strip() if '_TF:' in position.comment else "H1"
                                                tf_data_for_pos_mgmt = data['all_timeframe_data'].get(position_primary_tf, {})
                                                latest_candle_data_for_mgmt = tf_data_for_pos_mgmt.get('latest_candle_data', {})
                                                
                                                if latest_candle_data_for_mgmt: 
                                                    trade_manager.manage_open_position(
                                                        position, data['current_price'], data['symbol_info'], 
                                                        latest_candle_data_for_mgmt, config, log_activity
                                                    )
                                                else:
                                                    log_activity(f"Warning: No valid candle data for {position.symbol} for position management on TF {position_primary_tf}.", "yellow")
                                    else:
                                        current_cycle_all_symbols_data_dicts.append({
                                            'symbol': symbol_config["name"], 
                                            'current_price': 0, 'spread': 0, 'all_timeframe_data': {}, 
                                            'symbol_info': None, 'aggregated_buy_strength': 0, 
                                            'aggregated_sell_strength': 0, 
                                            'mtf_confluence_text': display_manager.Text("NO DATA", style="dim yellow")
                                        })
                            
                            # Update the main lists with the newly fetched data
                            all_symbols_data_dicts = current_cycle_all_symbols_data_dicts
                            trade_setups = current_cycle_trade_setups

                            # --- Display Update ---
                            # Format all data for display just once per refresh cycle
                            # all_symbols_display_lines = []
                            # for symbol_data_dict in all_symbols_data_dicts:
                            #     all_symbols_display_lines.extend(display_manager.format_data_for_display(symbol_data_dict))
                            #     all_symbols_display_lines.append(display_manager.Text("\n")) # Add a blank line between symbols

                            display_manager.render_full_display(
                                current_time.strftime('%Y-%m-%d %H:%M:%S'), 
                                total_symbols_to_monitor_count, 
                                all_symbols_data_dicts,  # Pass the list of dicts, not formatted lines
                                trade_setups, 
                                account_info, 
                                config
                            )
                            
                            # Process trade setups and execute trades
                            if trade_setups:
                                logging.info(f"Processing {len(trade_setups)} trade setups...")
                                trade_executor.process_trade_setups(trade_setups, account_info)
                            
                            last_table_refresh_time = time.time() # Reset refresh timer
                        
                        time.sleep(0.01) # Small sleep for CPU efficiency, adjust as needed.
                        
                    except KeyboardInterrupt:
                        logging.info("Received keyboard interrupt, shutting down gracefully")
                        break
                    except Exception as e:
                        logging.error(f"Error in monitoring loop: {str(e)}")
                        logging.error(traceback.format_exc())
                        time.sleep(60)  # Wait before retrying
                        continue
                        
            except Exception as e:
                logging.error(f"Critical error in main loop: {str(e)}")
                logging.error(traceback.format_exc())
                time.sleep(300)  # Wait 5 minutes before restarting
                continue
            finally:
                try:
                    mt5.shutdown()
                    logging.info("MT5 connection shut down")
                except:
                    pass
                
    except KeyboardInterrupt:
        logging.info("Shutting down trading system...")
    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")
    finally:
        # Cleanup
        if 'trade_executor' in locals():
            del trade_executor
        mt5.shutdown()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        logging.critical(traceback.format_exc())
        sys.exit(1)
