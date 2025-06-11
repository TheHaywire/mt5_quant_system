# trade_manager.py

import MetaTrader5 as mt5
import pandas as pd
import numpy as np 

def send_trade_order(setup_data: dict, trade_type: int, magic_number: int, live_trading_enabled: bool, log_activity_func):
    """
    Sends a market order to MT5.
    
    Args:
        setup_data (dict): Dictionary containing details of the trade setup.
        trade_type (int): mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL.
        magic_number (int): Unique ID for trades opened by this script.
        live_trading_enabled (bool): Flag to enable/disable live trade execution.
        log_activity_func (function): Function to log activities.

    Returns:
        mt5.TradeRequest: The result object from mt5.order_send, or a string for dry run.
    """
    symbol = setup_data['Symbol']
    lot = setup_data['Position Size']
    sl = setup_data['Stop Loss']
    tp = setup_data['Take Profit']
    price = setup_data['Entry'] 
    symbol_info = setup_data['symbol_info'] 

    if pd.isna(lot) or lot <= 0:
        log_activity_func(f"Cannot place trade for {symbol}: Invalid lot size ({lot:.2f}).", "red")
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20, 
        "magic": magic_number,
        "comment": f"QuantAI_Monitor_Trade_TF:{setup_data['Primary TF for Setup']}", 
        "type_time": mt5.ORDER_TIME_GTC, 
        "type_filling": mt5.ORDER_FILLING_FOC, 
    }

    if symbol_info:
        request["price"] = round(request["price"] / symbol_info.point) * symbol_info.point
        if request["sl"] != 0 and not pd.isna(request["sl"]): 
            request["sl"] = round(request["sl"] / symbol_info.point) * symbol_info.point
        if request["tp"] != 0 and not pd.isna(request["tp"]): 
            request["tp"] = round(request["tp"] / symbol_info.point) * symbol_info.point


    if live_trading_enabled:
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log_activity_func(f"ERROR sending {trade_type} order for {symbol}: {result.retcode} - {result.comment} (Deal {result.deal} Order {result.order})", "bold red")
            if result.request:
                log_activity_func(f"Request details: {result.request}", "red")
            return None
        else:
            log_activity_func(f"Successfully placed {trade_type} order for {symbol} ({lot:.2f} lots). Ticket: {result.order}. Price: {result.price_open:.{symbol_info.digits}f}.", "bold green")
            return result
    else:
        log_activity_func(f"DRY RUN: Would place {trade_type} order for {symbol} ({lot:.2f} lots). Entry: {price:.{symbol_info.digits}f}, SL: {sl:.{symbol_info.digits}f}, TP: {tp:.{symbol_info.digits}f}", "yellow")
        return "DRY_RUN_SUCCESS"

def manage_open_position(position: mt5.TradePosition, current_price: float, symbol_info: mt5.SymbolInfo, latest_candle_data: dict, config_module, log_activity_func): 
    """
    Manages an open position (e.g., move SL to breakeven, trailing stop).
    
    Args:
        position (mt5.TradePosition): The open position object.
        current_price (float): The current market price of the symbol.
        symbol_info (mt5.SymbolInfo): Symbol information for price precision.
        latest_candle_data (dict): Latest indicator data for the position's relevant timeframe (e.g., for ATR).
        config_module (module): The loaded config module (e.g., `config.py` module).
        log_activity_func (function): Function to log activities.
    """
    atr = latest_candle_data.get('atr')
    if pd.isna(atr) or atr <= 0:
        return

    modified = False
    new_sl = position.sl 
    new_tp = position.tp 
    
    price_point = symbol_info.point
    
    # 1. Break-Even Logic
    if position.type == mt5.POSITION_TYPE_BUY:
        profit_points = (current_price - position.price_open) / price_point
        breakeven_sl_target = position.price_open + (config_module.BREAK_EVEN_BUFFER_POINTS * price_point) 
        initial_risk_points = (position.price_open - position.sl) / price_point if position.sl > 0 else 0
        
        if initial_risk_points > 0 and profit_points >= initial_risk_points * config_module.BREAK_EVEN_PROFIT_FACTOR: 
            breakeven_sl_target = round(breakeven_sl_target / price_point) * price_point
            if new_sl < breakeven_sl_target: 
                new_sl = breakeven_sl_target
                modified = True
                log_activity_func(f"Position {position.ticket} ({position.symbol}): Moving SL to breakeven ({new_sl:.{symbol_info.digits}f}).", "green")

    elif position.type == mt5.POSITION_TYPE_SELL:
        profit_points = (position.price_open - current_price) / price_point
        breakeven_sl_target = position.price_open - (config_module.BREAK_EVEN_BUFFER_POINTS * price_point) 
        initial_risk_points = (position.sl - position.price_open) / price_point if position.sl > 0 else 0

        if initial_risk_points > 0 and profit_points >= initial_risk_points * config_module.BREAK_EVEN_PROFIT_FACTOR: 
            breakeven_sl_target = round(breakeven_sl_target / price_point) * price_point
            if new_sl > breakeven_sl_target or new_sl == 0: 
                new_sl = breakeven_sl_target
                modified = True
                log_activity_func(f"Position {position.ticket} ({position.symbol}): Moving SL to breakeven ({new_sl:.{symbol_info.digits}f}).", "green")

    # 2. Trailing Stop Logic (Simple ATR-based trailing stop)
    if position.type == mt5.POSITION_TYPE_BUY:
        current_profit_points = (current_price - position.price_open) / price_point
        
        if current_profit_points >= config_module.TRAILING_STOP_POINTS_INITIAL and \
           (new_sl < (current_price - (atr * price_point)) or new_sl <= position.price_open): 
            
            proposed_sl = current_price - (atr * price_point * 0.5) 
            proposed_sl = round(proposed_sl / price_point) * price_point

            if proposed_sl > new_sl:
                new_sl = proposed_sl
                modified = True
                log_activity_func(f"Position {position.ticket} ({position.symbol}): Trailing SL to {new_sl:.{symbol_info.digits}f}.", "green")

    elif position.type == mt5.POSITION_TYPE_SELL:
        current_profit_points = (position.price_open - current_price) / price_point
        
        if current_profit_points >= config_module.TRAILING_STOP_POINTS_INITIAL and \
           (new_sl > (current_price + (atr * price_point)) or new_sl >= position.price_open): 
            
            proposed_sl = current_price + (atr * price_point * 0.5)
            proposed_sl = round(proposed_sl / price_point) * price_point

            if proposed_sl < new_sl or new_sl == 0:
                new_sl = proposed_sl
                modified = True
                log_activity_func(f"Position {position.ticket} ({position.symbol}): Trailing SL to {new_sl:.{symbol_info.digits}f}.", "green")

    # Execute modification if needed and SL/TP values have actually changed
    if modified and (position.sl != new_sl or position.tp != new_tp):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": new_sl,
            "tp": new_tp,
            "magic": config_module.MAGIC_NUMBER, 
            "comment": position.comment, 
        }
        if config_module.LIVE_TRADING_ENABLED: 
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                log_activity_func(f"Error modifying position {position.ticket}: {result.retcode} - {result.comment}", "red")
            else:
                log_activity_func(f"Modified position {position.ticket} ({position.symbol}) to SL: {new_sl:.{symbol_info.digits}f}, TP: {new_tp:.{symbol_info.digits}f}.", "green")
        else:
            log_activity_func(f"DRY RUN: Would modify position {position.ticket}. New SL: {new_sl:.{symbol_info.digits}f}, New TP: {new_tp:.{symbol_info.digits}f}", "yellow")

