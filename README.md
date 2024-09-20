# Trading Bot

A Python-based trading bot that uses Alpaca API, YFinance, VADER sentiment analysis, and OpenAI's GPT-3.5 to analyze financial data, execute trades, and manage risk. This bot is designed to operate automatically and includes several features such as risk management, news sentiment analysis, and portfolio management.

**Note**: This code is not public at the moment but may be in the future.

## Features

- **Automatic Trading**: The bot uses Alpaca API to trade stocks automatically.
- **RSI & Sentiment Analysis**: It utilizes RSI (Relative Strength Index) from `ta-lib` and sentiment analysis using VADER to make informed trading decisions.
- **GPT-3.5 Integration**: Queries OpenAI GPT-3.5 to get trade decisions based on stock data (RSI, sentiment, P/E ratio, VaR).
- **Risk Management**: Implements a maximum drawdown limit, maximum allocation per stock, and trailing stops.
- **Logging**: Trades and OpenAI decisions are logged into JSON files (`chat_gpt_logs.json` and `trade_decisions.json`), and committed periodically to a remote repository.
- **Email Notifications**: Sends error notifications via email using SMTP.
- **Portfolio Management**: Automatically liquidates the portfolio if its value falls below a defined threshold.
- **Real-time Stock Data**: Fetches real-time stock prices via Alpaca's WebSocket streaming API.

## Installation

1. Clone the repository (Note: The code is currently not public, but may be in the future).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install TA-Lib from source (if required):
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential wget
   wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   tar -xzf ta-lib-0.4.0-src.tar.gz
   cd ta-lib
   ./configure --prefix=/usr
   make
   sudo make install
   ```

## Usage

To run the bot:

```bash
python bot.py
```

The bot operates by streaming real-time stock price data, evaluating stocks using technical indicators, and querying GPT-3.5 for decisions.

### Features Overview

- **Sentiment Analysis**: Fetches recent news articles using Finnhub API and scores the sentiment using VADER Sentiment Analysis.
- **GPT-3.5 Trading Decision**: The bot collects the RSI value, sentiment score, P/E ratio, and VaR, and queries OpenAI to decide whether to buy or sell stocks.
- **Risk Management**: A max drawdown limit of 8% and a trailing stop of 2% are enforced to protect against significant losses.

### Auto Liquidation

The bot is designed to liquidate all positions at 3:50 PM Eastern Time and shut down to prevent further trading after the market closes.

## Logging and Commit

The bot logs its trading decisions and GPT queries to JSON files and periodically commits the logs to the GitHub repository (if configured).

```bash
chat_gpt_logs.json  # Logs GPT-based decisions
trade_decisions.json  # Logs all trade decisions
```

## Customization

You can modify the following parameters in the script:

- **Portfolio Settings**:
  - `INITIAL_PORTFOLIO_VALUE`: Initial portfolio value ($26,000).
  - `MIN_PORTFOLIO_VALUE`: Minimum portfolio value ($25,100).
  - `MAX_RISK_PER_TRADE`: Max risk per trade (1% of portfolio value).
  - `MAX_DRAW_DOWN_LIMIT`: Maximum allowable drawdown (8%).

- **Alpaca API Settings**:
  - Replace the Alpaca API credentials with your own in the script:
    ```python
    API_KEY = 'YOUR_ALPACA_API_KEY'
    API_SECRET = 'YOUR_ALPACA_API_SECRET'
    BASE_URL = 'https://paper-api.alpaca.markets'
    ```

- **Email Notifications**: Configure the email notifications by providing your email credentials:
  ```python
  EMAIL_USER = 'your-email@gmail.com'
  EMAIL_PASS = 'your-email-password'
  ```

## Contributing

At this time, contributions are not accepted because the code is not public. However, this may change in the future.

## License

This project is licensed under the MIT License.


