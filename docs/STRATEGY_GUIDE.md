# Trading Strategy Guide

## Overview

The MT5 Quant Trading System implements a multi-timeframe, multi-indicator strategy that combines technical analysis with risk management to identify high-probability trading setups. This document explains the strategy components, signal generation, and trade management.

## Strategy Components

### 1. Multi-Timeframe Analysis

The system analyzes multiple timeframes to identify trade setups:
- **M1**: Entry timing and short-term momentum
- **M5**: Initial trend confirmation
- **M15**: Primary trading timeframe
- **H1**: Trend direction
- **H4**: Major trend context
- **D1**: Long-term bias

### 2. Technical Indicators

#### Primary Indicators
- **RSI (Relative Strength Index)**
  - Period: 14
  - Overbought: 70
  - Oversold: 30
  - Divergence detection

- **MACD (Moving Average Convergence Divergence)**
  - Fast Period: 12
  - Slow Period: 26
  - Signal Period: 9
  - Histogram analysis

- **EMA (Exponential Moving Average)**
  - Short-term: 9
  - Medium-term: 21
  - Long-term: 50
  - Crossover signals

- **Bollinger Bands**
  - Period: 20
  - Standard Deviation: 2
  - Volatility measurement
  - Price channel analysis

#### Secondary Indicators
- **Volume Analysis**
  - Volume trend
  - Volume-price relationship
  - Unusual volume detection

- **Support/Resistance**
  - Dynamic levels
  - Price action patterns
  - Breakout confirmation

## Signal Generation

### 1. Entry Signals

#### Bullish Setup
- RSI oversold (< 30) with bullish divergence
- MACD histogram turning positive
- Price above EMA(50)
- Volume confirmation
- Minimum 1.5:1 risk-reward ratio

#### Bearish Setup
- RSI overbought (> 70) with bearish divergence
- MACD histogram turning negative
- Price below EMA(50)
- Volume confirmation
- Minimum 1.5:1 risk-reward ratio

### 2. Signal Strength Calculation

```python
def calculate_signal_strength(data):
    # RSI Component (30%)
    rsi_score = calculate_rsi_score(data['rsi'])
    
    # MACD Component (30%)
    macd_score = calculate_macd_score(data['macd'])
    
    # EMA Component (20%)
    ema_score = calculate_ema_score(data['ema'])
    
    # Volume Component (20%)
    volume_score = calculate_volume_score(data['volume'])
    
    # Weighted Sum
    total_score = (
        0.30 * rsi_score +
        0.30 * macd_score +
        0.20 * ema_score +
        0.20 * volume_score
    )
    
    return total_score
```

### 3. Trade Setup Validation

```python
def validate_trade_setup(signal, data):
    # Minimum Requirements
    if signal['strength'] < 0.7:  # 70% minimum confidence
        return False
        
    if signal['rr_ratio'] < 1.5:  # Minimum 1.5:1 risk-reward
        return False
        
    if not check_correlation_limits(data['symbol']):
        return False
        
    return True
```

## Position Management

### 1. Entry Rules
- Wait for signal confirmation
- Verify risk parameters
- Calculate position size
- Place entry order with stop-loss

### 2. Stop-Loss Management
- Initial stop: 2x ATR from entry
- Break-even: Move to entry after 1:1 risk-reward
- Trailing stop: 1.5x ATR trailing

### 3. Take-Profit Levels
- TP1: 1.5x risk (50% of position)
- TP2: 2x risk (remaining position)
- Trailing stop after TP1

## Risk Management

### 1. Position Sizing
```python
def calculate_position_size(account_equity, risk_per_trade, stop_distance):
    risk_amount = account_equity * risk_per_trade
    position_size = risk_amount / stop_distance
    return round(position_size, 2)  # Round to 2 decimal places
```

### 2. Exposure Limits
- Maximum 1% risk per trade
- Maximum 5% total exposure
- Maximum 2 positions per symbol
- Correlation-based limits

### 3. Drawdown Protection
- Maximum 20% drawdown
- Reduce position size by 50% at 15% drawdown
- Stop trading at 20% drawdown

## Performance Optimization

### 1. Strategy Parameters
- Optimize indicator periods
- Adjust signal thresholds
- Fine-tune risk parameters
- Test different timeframes

### 2. Backtesting
- Use historical data
- Validate strategy assumptions
- Measure performance metrics
- Optimize parameters

### 3. Live Testing
- Start with small position sizes
- Monitor strategy behavior
- Adjust parameters gradually
- Track performance metrics

## Common Scenarios

### 1. Strong Trend
- Multiple timeframe alignment
- High volume confirmation
- Clear support/resistance
- Extended profit targets

### 2. Range-Bound Market
- Tighter stop-losses
- Reduced position sizes
- Focus on mean reversion
- Quick profit taking

### 3. High Volatility
- Wider stop-losses
- Reduced position sizes
- Focus on momentum
- Avoid trading during news

## Best Practices

1. **Trade Management**
   - Stick to the strategy
   - Follow risk management rules
   - Keep detailed trade logs
   - Review performance regularly

2. **Risk Control**
   - Never exceed position limits
   - Monitor correlation exposure
   - Maintain stop-loss discipline
   - Protect account equity

3. **System Maintenance**
   - Regular parameter review
   - Performance analysis
   - Strategy optimization
   - Documentation updates 