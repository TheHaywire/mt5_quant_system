# display_manager.py

import os
from datetime import datetime
from collections import deque
from rich.console import Console # Keep console for basic print, but no advanced rendering
from rich.text import Text # Keep Text for log_activity internal storage and converting to plain
import pandas as pd 
import numpy as np 
import MetaTrader5 as mt5 
from rich.table import Table
from rich.panel import Panel
from rich import box

# Initialize Rich Console (shared instance) - used only for basic print() fallback
console = Console()

# --- Global Activity Log ---
# This deque will store recent activity messages for display.
# It uses Rich Text objects internally to retain style info, but we'll print .plain
activity_log = deque(maxlen=10) 

def log_activity(message: str, style: str = "white"):
    """Adds a timestamped message to the activity log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Store as Rich Text to keep track of style, even if we print plain later
    activity_log.append(Text(f"[{timestamp}] {message}", style=style))

def clear_terminal():
    """Clears the terminal screen using OS command."""
    os.system('cls' if os.name == 'nt' else 'clear')

# --- Fixed display widths for universal compatibility ---
CONTENT_WIDTH = 90 # Width for content inside the borders
TOTAL_LINE_WIDTH = CONTENT_WIDTH + 4 # 2 for side borders '│', 2 for spaces ' '

def _pad_content_line(line_content: str, align: str = 'left') -> str:
    """Pads a plain string to CONTENT_WIDTH with alignment."""
    # Ensure the content is treated as plain text for padding calculations
    plain_content = Text(line_content).plain # Strip rich markup for accurate length
    if align == 'center':
        return plain_content.center(CONTENT_WIDTH)
    elif align == 'right':
        return plain_content.rjust(CONTENT_WIDTH)
    return plain_content.ljust(CONTENT_WIDTH)

def render_section_header(title: str, style: str = "white"):
    """Renders a section header with ASCII borders and title."""
    header_line = f" {title} ".center(CONTENT_WIDTH, '─')
    print(Text(f"╭{header_line}╮", style=style))

def render_section_footer(style: str = "white"):
    """Renders a section footer with ASCII borders."""
    print(Text(f"╰{'─' * CONTENT_WIDTH}╯", style=style))

def render_line(content: str, style: str = "white", align: str = 'left'):
    """Renders a single line within a section, with padding and borders."""
    padded_content = _pad_content_line(content, align)
    print(Text(f"│ {padded_content} │", style=style))

def render_empty_line():
    """Renders an empty line within a section for spacing."""
    print(f"│{' ' * CONTENT_WIDTH}│")


def format_data_for_display(data: dict): 
    """
    Formats relevant data for a single symbol's display.
    Returns a list of plain strings with Rich style markers.
    """
    formatted_lines = []

    # Handle case where data might be a string (placeholder data)
    if isinstance(data, str):
        formatted_lines.append(f"[dim]{data}[/dim]")
        return formatted_lines

    def get_val_fmt(data_dict, key, fmt_str=None, default='N/A'):
        if not isinstance(data_dict, dict):
            return default
        val = data_dict.get(key, np.nan)
        if pd.isna(val) or val is None: 
            return default
        if fmt_str:
            return fmt_str.format(val)
        return str(val)

    primary_tf_for_display = "M1"
    m1_data = data.get('all_timeframe_data', {}).get(primary_tf_for_display, {}) if isinstance(data, dict) else {}

    if not m1_data.get('indicators_complete', False):
        for tf_name in ["M5", "M15", "H1", "H4", "D1"]: 
            if isinstance(data, dict) and data.get('all_timeframe_data', {}).get(tf_name, {}).get('indicators_complete', False):
                m1_data = data['all_timeframe_data'][tf_name]
                primary_tf_for_display = tf_name 
                break

    indicators_complete_for_display_tf = m1_data.get('indicators_complete', False)
    latest = m1_data.get('latest_candle_data', {}) if isinstance(m1_data, dict) else {}

    symbol_text_plain = data.get('symbol', 'N/A') if isinstance(data, dict) else 'N/A'
    formatted_lines.append(f"Symbol: [bold cyan]{symbol_text_plain}[/bold cyan]") 

    if not indicators_complete_for_display_tf:
        formatted_lines.append(f"  [red]NO DATA AVAILABLE FOR INDICATORS[/red]")
        formatted_lines.append(f"  Price: {get_val_fmt(data, 'current_price', '{:.5f}')}")
        formatted_lines.append(f"  Spread: {get_val_fmt(data, 'spread', '{:.1f}')}")
        formatted_lines.append(f"  Signal: [dim yellow]NO DATA[/dim yellow]")
        formatted_lines.append(f"  Sentiment: N/A")
        formatted_lines.append(f"  MTF Confluence: N/A")
    else:
        buy_strength = data.get('aggregated_buy_strength', 0) if isinstance(data, dict) else 0
        sell_strength = data.get('aggregated_sell_strength', 0) if isinstance(data, dict) else 0

        signal_text = "NEUTRAL"
        signal_style = "white"
        sentiment_text = "Neutral"
        sentiment_style = "white"

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
        
        price_fmt_str = "{:.5f}" 
        if isinstance(data, dict) and data.get('symbol_info') and hasattr(data['symbol_info'], 'digits'):
            price_fmt_str = f"{{:.{data['symbol_info'].digits}f}}"

        formatted_lines.append(f"  Price: {get_val_fmt(data, 'current_price', price_fmt_str)}")
        formatted_lines.append(f"  Spread: {get_val_fmt(data, 'spread', '{:.1f}')}")
        formatted_lines.append(f"  RSI: {get_val_fmt(latest, 'rsi', '{:.1f}')}")
        formatted_lines.append(f"  MACD Hist: {get_val_fmt(latest, 'macd_hist', '{:.5f}')}")
        
        ema_trend_str = "N/A"
        ema_trend_style = "white"
        ema_fast = latest.get('ema_fast')
        ema_slow = latest.get('ema_slow')
        if not (pd.isna(ema_fast) or pd.isna(ema_slow)): 
            ema_diff = ema_fast - ema_slow
            if ema_diff > 0: ema_trend_str = "BULL"
            elif ema_diff < 0: ema_trend_str = "BEAR"
            else: ema_trend_str = "FLAT"
        formatted_lines.append(f"  EMA Trend: [{ema_trend_style}]{ema_trend_str}[/{ema_trend_style}]")

        ma_trend_str = "N/A"
        ma_trend_style = "white"
        sma_50 = latest.get('sma_50')
        sma_200 = latest.get('sma_200')
        if not (pd.isna(sma_50) or pd.isna(sma_200)): 
            if sma_50 > sma_200: ma_trend_str = "BULL"
            elif sma_50 < sma_200: ma_trend_str = "BEAR"
            else: ma_trend_str = "FLAT"
        formatted_lines.append(f"  MA Trend: [{ma_trend_style}]{ma_trend_str}[/{ma_trend_style}]")
        
        bb_position_str = "N/A"
        close_price = latest.get('close')
        bb_upper = latest.get('bb_upper')
        bb_lower = latest.get('bb_lower')

        if not (pd.isna(close_price) or pd.isna(bb_upper) or pd.isna(bb_lower)) and (bb_upper - bb_lower) != 0: 
            bb_pos = (close_price - bb_lower) / (bb_upper - bb_lower)
            bb_position_str = f"{bb_pos:.0%}"
        formatted_lines.append(f"  BB Pos: {bb_position_str}")

        formatted_lines.append(f"  Buy Str: {get_val_fmt({'strength': buy_strength}, 'strength', '{:.0%}')}")
        formatted_lines.append(f"  Sell Str: {get_val_fmt({'strength': sell_strength}, 'strength', '{:.0%}')}")
        formatted_lines.append(f"  Signal: [{signal_style}]{signal_text}[/{signal_style}]")
        formatted_lines.append(f"  Sentiment: [{sentiment_style}]{sentiment_text}[/{sentiment_style}]")
        
        mtf_confluence_status_text = data.get('mtf_confluence_text', Text("N/A", style="dim")) if isinstance(data, dict) else Text("N/A", style="dim")
        formatted_lines.append(f"  MTF Confluence: {mtf_confluence_status_text.plain if hasattr(mtf_confluence_status_text, 'plain') else str(mtf_confluence_status_text)}") 
    
    return formatted_lines

def format_setup_for_display(setup: dict, account_equity: float, config_module):
    """
    Formats a single high probability setup for display.
    Returns a list of plain strings with Rich style markers.
    """
    setup_lines = []
    
    setup_signal_style = "green" if setup['Signal'] == "BUY" else "red"
    
    setup_lines.append(f"\n[bold underline]{setup['Symbol']}[/bold underline] ({setup['Primary TF for Setup']})")
    setup_lines.append(f"  Signal: [{setup_signal_style}]{setup['Signal']}[/{setup_signal_style}]")
    
    price_digits = 5 
    if setup.get('symbol_info') and hasattr(setup['symbol_info'], 'digits'):
        price_digits = setup['symbol_info'].digits 
    
    setup_lines.append(f"  Entry: {setup['Entry']:.{price_digits}f}")
    setup_lines.append(f"  Stop Loss: {setup['Stop Loss']:.{price_digits}f}")
    setup_lines.append(f"  Take Profit: {setup['Take Profit']:.{price_digits}f}")
    
    setup_lines.append(f"  Position Size: {setup['Position Size']:.2f} lots")
    setup_lines.append(f"  Risk/Reward: {setup['Risk/Reward']:.2f}")
    setup_lines.append(f"  Setup Confidence: [cyan]{setup['Setup Confidence']:.2f}[/cyan]") 
    setup_lines.append(f"  Risk Amount: [red]${setup['Risk Amount']:.2f}[/red]")
    setup_lines.append(f"  Potential Profit: [green]${setup['Potential Profit']:.2f}[/green]")
    
    setup_lines.append(f"\n  [bold]Risk/Position Sizing Rationale:[/bold]")
    setup_lines.append(f"    Position size is calculated to risk [bold]{config_module.RISK_PER_TRADE_PERCENT:.1%}[/bold] of your current equity (${account_equity:.2f}) if the Stop Loss is hit.")
    setup_lines.append(f"    This ensures consistent risk management per trade, automatically adjusting lot size as your equity changes.")
    setup_lines.append(f"    Automated lot size clamped between [bold]{config_module.MIN_LOT_SIZE_CLI:.2f}[/bold] and [bold]{config_module.MAX_LOT_SIZE_CLI:.2f}[/bold] lots.")

    setup_lines.append(f"\n  [bold]Market Conditions:[/bold]")
    setup_lines.append(f"    Last Candle: {setup['Last Candle']}")
    setup_lines.append(f"    Candle Size: {setup['Candle Size']:.4f}")
    setup_lines.append(f"    Volume vs Average: {setup['Volume vs Average']:.2f}x")
    setup_lines.append(f"    Trend Strength (ADX): {setup['Trend Strength']:.1f}")
    setup_lines.append(f"    RSI: {setup['RSI']:.1f}") 
    
    return setup_lines

def create_market_data_table(all_symbols_data_dicts: list) -> Table:
    """Creates a table for market data display."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    
    # Add columns
    table.add_column("Symbol", style="cyan", width=10)
    table.add_column("Price", justify="right", width=12)
    table.add_column("Spread", justify="right", width=8)
    table.add_column("RSI", justify="right", width=8)
    table.add_column("MACD", justify="right", width=12)
    table.add_column("Trend", width=12)
    table.add_column("BB%", justify="right", width=8)
    table.add_column("Signal", width=12)
    table.add_column("MTF", width=15)

    for data in all_symbols_data_dicts:
        if isinstance(data, str):
            continue

        symbol = data.get('symbol', 'N/A')
        price = data.get('current_price', 0)
        spread = data.get('spread', 0)
        
        # Get latest data
        latest = data.get('all_timeframe_data', {}).get('M1', {}).get('latest_candle_data', {})
        if not latest:
            for tf in ['M5', 'H1', 'H4', 'D1']:
                latest = data.get('all_timeframe_data', {}).get(tf, {}).get('latest_candle_data', {})
                if latest:
                    break

        rsi = latest.get('rsi', 0)
        macd = latest.get('macd_hist', 0)
        
        # Determine trend
        ema_trend = "N/A"
        if not pd.isna(latest.get('ema_fast')) and not pd.isna(latest.get('ema_slow')):
            ema_trend = "BULL" if latest['ema_fast'] > latest['ema_slow'] else "BEAR"
        
        # BB Position
        bb_pos = "N/A"
        if all(k in latest for k in ['close', 'bb_upper', 'bb_lower']):
            if latest['bb_upper'] != latest['bb_lower']:
                bb_pos = f"{((latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower']) * 100):.0f}%"

        # Signal and MTF
        signal = "NEUTRAL"
        signal_style = "white"
        if data.get('aggregated_buy_strength', 0) >= 0.7:
            signal = "STRONG BUY"
            signal_style = "bold green"
        elif data.get('aggregated_sell_strength', 0) >= 0.7:
            signal = "STRONG SELL"
            signal_style = "bold red"
        
        mtf = data.get('mtf_confluence_text', Text("N/A", style="dim"))
        mtf_text = mtf.plain if hasattr(mtf, 'plain') else str(mtf)

        # Add row with styling
        table.add_row(
            symbol,
            f"{price:.5f}",
            f"{spread:.1f}",
            f"{rsi:.1f}",
            f"{macd:.5f}",
            ema_trend,
            bb_pos,
            Text(signal, style=signal_style),
            mtf_text
        )

    return table

def create_setups_table(trade_setups: list, account_equity: float, config_module) -> Table:
    """Creates a table for trade setups display."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold yellow")
    
    # Add columns
    table.add_column("Symbol", style="cyan", width=10)
    table.add_column("TF", width=5)
    table.add_column("Signal", width=8)
    table.add_column("Entry", justify="right", width=10)
    table.add_column("SL", justify="right", width=10)
    table.add_column("TP", justify="right", width=10)
    table.add_column("Size", justify="right", width=8)
    table.add_column("R/R", justify="right", width=6)
    table.add_column("Conf", justify="right", width=6)
    table.add_column("Risk $", justify="right", width=10)

    for setup in trade_setups:
        signal_style = "green" if setup['Signal'] == "BUY" else "red"
        
        table.add_row(
            setup['Symbol'],
            setup['Primary TF for Setup'],
            Text(setup['Signal'], style=signal_style),
            f"{setup['Entry']:.5f}",
            f"{setup['Stop Loss']:.5f}",
            f"{setup['Take Profit']:.5f}",
            f"{setup['Position Size']:.2f}",
            f"{setup['Risk/Reward']:.2f}",
            f"{setup['Setup Confidence']:.2f}",
            f"${setup['Risk Amount']:.2f}"
        )

    return table

def create_positions_table(positions: list) -> Table:
    """Creates a table for open positions display."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
    
    # Add columns
    table.add_column("Symbol", style="cyan", width=10)
    table.add_column("Type", width=6)
    table.add_column("Size", justify="right", width=8)
    table.add_column("Entry", justify="right", width=10)
    table.add_column("Current", justify="right", width=10)
    table.add_column("P/L", justify="right", width=12)

    for pos in positions:
        pos_type = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        pos_style = "green" if pos.profit >= 0 else "red"
        
        table.add_row(
            pos.symbol,
            Text(pos_type, style="green" if pos_type == "BUY" else "red"),
            f"{pos.volume:.2f}",
            f"{pos.price_open:.5f}",
            f"{pos.price_current:.5f}",
            Text(f"${pos.profit:.2f}", style=pos_style)
        )

    return table

def render_full_display(current_time_str: str, total_symbols_to_monitor_count: int, all_symbols_data_dicts: list, trade_setups: list, account_info, config_module):
    """Renders the entire console display using Rich tables."""
    clear_terminal()

    # Header
    console.print(Panel(f"QuantAI Monitor - Live Market Overview\n{current_time_str}", 
                       style="bold", border_style="cyan"))
    console.print()

    # Market Data Table
    console.print(Panel("Market Data Snapshot", style="bold green"))
    market_table = create_market_data_table(all_symbols_data_dicts)
    console.print(market_table)
    console.print()

    # Trade Setups Table
    console.print(Panel("High Probability Setups", style="bold yellow"))
    if trade_setups:
        setups_table = create_setups_table(trade_setups, account_info.equity, config_module)
        console.print(setups_table)
    else:
        console.print("[dim]No high probability setups identified at this time.[/dim]")
    console.print()

    # Status Panel
    status_table = Table(box=box.ROUNDED, show_header=False)
    status_table.add_column("Metric", style="bold")
    status_table.add_column("Value")
    
    status_table.add_row("Last Updated", current_time_str)
    status_table.add_row("Total Symbols", str(total_symbols_to_monitor_count))
    status_table.add_row("High Probability Setups", str(len(trade_setups)))
    
    # Open Positions
    current_positions = mt5.positions_get(magic=config_module.MAGIC_NUMBER)
    if current_positions:
        status_table.add_row("Open Positions", str(len(current_positions)))
        console.print(Panel("Open Positions", style="bold blue"))
        positions_table = create_positions_table(current_positions)
        console.print(positions_table)
    else:
        status_table.add_row("Open Positions", "None")
    
    # Activity Log
    console.print("\n[bold]Recent Activity Log:[/bold]")
    for log_entry in activity_log:
        console.print(log_entry)
    
    console.print("\n[dim]Press Ctrl+C to exit.[/dim]")

