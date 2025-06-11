# Quick Start Guide

This guide will help you set up and run the MT5 Quant Trading System quickly.

## Prerequisites

1. **Windows 11 x64**
2. **Python 3.11+**
   - Download from [Python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

3. **MetaTrader 5**
   - Download from [MetaQuotes](https://www.metatrader5.com/en/download)
   - Install and create a demo account
   - Note down your account credentials

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/TheHaywire/mt5_quant_system.git
   cd mt5_quant_system
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure MT5**
   - Open `config.py`
   - Update MT5 path:
     ```python
     MT5_PATH = "C:/Program Files/MetaTrader 5/terminal64.exe"  # Update this path
     ```
   - Update account credentials:
     ```python
     MT5_LOGIN = "your_account_number"
     MT5_PASSWORD = "your_password"
     MT5_SERVER = "your_broker_server"
     ```

## First Run

1. **Start MT5 Terminal**
   - Launch MetaTrader 5
   - Log in to your account
   - Keep it running in the background

2. **Run the System**
   ```bash
   python main.py
   ```

3. **Monitor the Console**
   - Market data display
   - Trade signals
   - Position updates
   - System status

## Configuration Guide

### Trading Parameters
Edit `config.py` to customize:

```python
# Trading Symbols
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

# Risk Parameters
RISK_PER_TRADE = 0.01  # 1% risk per trade
MAX_DRAWDOWN = 0.20    # 20% maximum drawdown

# Strategy Parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
```

### Timeframes
The system uses multiple timeframes:
- M1 (1 minute)
- M5 (5 minutes)
- M15 (15 minutes)
- H1 (1 hour)
- H4 (4 hours)
- D1 (1 day)

## System Features

### 1. Market Data Display
- Real-time price updates
- Technical indicators
- Signal strength
- Trade setups

### 2. Trade Management
- Automated entry/exit
- Stop-loss management
- Take-profit levels
- Position sizing

### 3. Risk Management
- Per-trade risk limits
- Maximum drawdown protection
- Correlation-based exposure
- Position limits

## Common Issues

### 1. MT5 Connection
- Ensure MT5 is running
- Verify account credentials
- Check internet connection
- Restart MT5 if needed

### 2. Python Errors
- Verify Python version (3.11+)
- Check virtual environment
- Update dependencies
- Check error logs

### 3. Trading Issues
- Verify account balance
- Check symbol availability
- Monitor margin requirements
- Review trade logs

## Next Steps

1. **Review Documentation**
   - Read `SYSTEM_ARCHITECTURE.md`
   - Study `STRATEGY_GUIDE.md`
   - Check `docs/` for more details

2. **Start Small**
   - Use demo account
   - Start with minimum risk
   - Monitor performance
   - Adjust parameters

3. **Regular Maintenance**
   - Check system logs
   - Monitor performance
   - Update parameters
   - Backup configuration

## Support

For issues and questions:
1. Check the documentation
2. Review error logs
3. Create a GitHub issue
4. Contact support

## Backup

Use the backup script:
```bash
.\scripts\backup_system.ps1
```

This creates a timestamped backup in the `backups/` directory.

## Updates

To update the system:
```bash
git pull origin main
pip install -r requirements.txt
```

## Security Notes

1. **Never share your:**
   - MT5 credentials
   - API keys
   - Account details

2. **Always:**
   - Use demo account for testing
   - Start with small position sizes
   - Monitor system performance
   - Keep backups of your configuration 