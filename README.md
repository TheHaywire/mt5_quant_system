# MT5 Quant Trading System

A production-grade, Windows-native, fully-automated MetaTrader 5 trading system with multi-timeframe analysis, risk management, and real-time monitoring.

## Features

- **Multi-Timeframe Analysis**
  - M1, M5, M15, H1, H4, D1 timeframe support
  - Weighted signal aggregation
  - Trend confluence detection

- **Risk Management**
  - Kelly-capped position sizing
  - Dynamic lot calculation
  - Maximum drawdown protection
  - Multi-symbol correlation handling

- **Trade Execution**
  - Automated entry/exit
  - Break-even management
  - Trailing stops
  - Slippage control

- **Real-time Monitoring**
  - Live market data display
  - Trade setup alerts
  - Position management
  - System health metrics

## Requirements

- Windows 11 x64
- Python 3.11
- MetaTrader 5 Terminal
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mt5_quant_system.git
cd mt5_quant_system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure MT5:
   - Install MetaTrader 5
   - Copy terminal64.exe path to config.py
   - Update login credentials in config.py

## Configuration

Edit `config.py` to set:
- MT5 connection details
- Risk parameters
- Trading symbols
- Timeframe settings
- Strategy thresholds

## Usage

1. Start MT5 terminal
2. Run the system:
```bash
python main.py
```

3. Monitor the console for:
   - Market data
   - Trade setups
   - Position updates
   - System status

## Project Structure

```
mt5_quant_system/
├── core/                    # Core trading logic
│   ├── mt5_connector.py     # MT5 connection handler
│   ├── trade_executor.py    # Trade execution
│   └── risk_manager.py      # Risk management
├── scripts/                 # Utility scripts
├── logs/                    # Trading logs
├── config.py               # Configuration
├── main.py                 # Main entry point
├── strategy.py             # Trading strategy
├── display_manager.py      # Console UI
└── requirements.txt        # Dependencies
```

## Risk Warning

This system is for educational purposes. Trading involves significant risk. Always:
- Test thoroughly on demo accounts
- Start with small position sizes
- Monitor system performance
- Keep backups of your configuration

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues and feature requests, please use the GitHub issue tracker. 