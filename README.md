# TradingBot-Public

# Trading Bot Workflow

This repository contains a fully automated trading bot built using Alpaca's API, OpenAI's GPT, and additional financial analysis tools such as `yfinance`, `ta-lib`, and sentiment analysis via VADER. The bot uses a combination of real-time stock price analysis, sentiment analysis, and risk management techniques to make trading decisions.

## Features

- **Real-Time Trading**: Subscribes to live stock price data for selected S&P 500 stocks (e.g., AAPL, MSFT, GOOGL, AMZN, TSLA).
- **Sentiment Analysis**: Uses the VADER sentiment analyzer to score financial news articles and influence trading decisions.
- **Technical Analysis**: Implements RSI (Relative Strength Index) for momentum trading.
- **Value at Risk (VaR)**: Estimates the potential loss of an investment over a specific period of time.
- **Risk Management**: Automatically calculates position sizes based on portfolio size, and limits drawdown.
- **Automated Liquidation**: Automatically liquidates all positions at 3:50 PM EST to avoid holding overnight.
- **GitHub Integration**: Commits log files to a GitHub repository for tracking trade decisions and performance.

## Workflow

This bot runs on a GitHub Actions workflow that:

1. Installs necessary dependencies, including Python packages and `ta-lib`.
2. Runs the trading bot script.
3. Commits logs to the repository periodically and at the end of the trading session.

### Scheduled Workflow

The bot is triggered by a cron job that runs at:

- **9:25 AM EST** (13:25 UTC during Daylight Savings)
- **9:25 AM EST** (14:25 UTC during Standard Time)

The job will run until 3:50 PM EST, at which point it will liquidate all positions and wait for 10 minutes before shutting down.

## Getting Started

### Prerequisites

- **Python 3.12.6** (or the version specified in `setup-python`)
- **Alpaca Paper API** (Create an account at [Alpaca](https://alpaca.markets))
- **OpenAI API** (Create an account at [OpenAI](https://openai.com))
- **GitHub Repository** (for logging trade decisions)
- **News API** (such as [Finnhub](https://finnhub.io/))

### Environment Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/YOUR_GITHUB_USERNAME/TradingBot.git
    cd TradingBot
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up Alpaca, OpenAI, and News API keys in the script:
    - `API_KEY`: Alpaca API key
    - `API_SECRET`: Alpaca secret key
    - `openai.api_key`: OpenAI API key
    - `NEWS_API_KEY`: Finnhub (or other) News API key

4. Run the bot manually (optional):
    ```bash
    python bot.py
    ```

### GitHub Actions Integration

This bot is designed to run automatically via GitHub Actions. The `.github/workflows/main.yml` file defines the following steps:

- Checkout the repository.
- Install Python and dependencies.
- Run the bot script.
- Commit and push the trade log and decisions.

#### Workflow Triggers

- **Scheduled**: Runs the bot automatically based on the defined cron schedule (9:25 AM EST).
- **Manual**: The workflow can be manually triggered from the Actions tab.

### Logs

The bot logs trading decisions and API interactions in JSON format. Logs are committed to the GitHub repository:

- `chat_gpt_logs.json`: Logs interactions with OpenAI for trading decisions.
- `trade_decisions.json`: Logs the buy/sell decisions for each stock.

## Example Cron Schedule

```yaml
on:
  schedule:
    - cron: '25 13 * * 1-5'  # Daylight Savings
    - cron: '25 14 * * 1-5'  # Standard Time
  workflow_dispatch:
