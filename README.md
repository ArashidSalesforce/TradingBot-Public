
# Trading Bot: An Automated AI-Powered Financial Trader

This project is a comprehensive trading bot designed for automated trading using Alpaca's API, OpenAI for decision-making, and various financial tools and APIs. It makes informed trading decisions based on technical analysis, news sentiment, risk management strategies, and portfolio tracking.

**Note**: This code is not public at the moment but may be in the future.

## Features

### Real-Time Trading and Market Data
- **Market Data Feeds**: The bot subscribes to real-time stock price data using Alpaca's WebSocket streams.
- **Supported Stocks**: By default, the bot evaluates selected S&P 500 stocks (AAPL, MSFT, GOOGL, AMZN, TSLA) but can be configured for other stocks.

### Sentiment and Technical Analysis
- **VADER Sentiment Analysis**: Fetches real-time news sentiment from sources like Finnhub to influence trade decisions.
- **RSI Indicator**: Uses the Relative Strength Index (RSI) for momentum trading and detecting overbought/oversold conditions.
- **P/E Ratio**: Integrates stock valuation data to assess over- or undervalued stocks.
- **Value at Risk (VaR)**: Computes potential portfolio losses based on historical stock price movements.

### Risk Management
- **Max Risk per Trade**: Automatically calculates position size to limit the risk to 1% of the portfolio value per trade.
- **Drawdown Protection**: Automatically liquidates positions if the portfolio drawdown exceeds a predefined threshold of 8%.
- **Trailing Stop-Losses**: Sets trailing stops to protect gains after profitable trades.
- **Portfolio Liquidation**: All positions are liquidated at 3:50 PM EST to avoid holding overnight risk.

### AI-Driven Decision Making
- **OpenAI Integration**: The bot uses OpenAI's GPT-3.5 for decision making. It combines sentiment, RSI, P/E ratio, and VaR to get suggestions on whether to buy or sell stocks, based on market conditions.

### GitHub Integration and Logs
- **Auto Commit Logs**: The bot periodically commits trading logs (`chat_gpt_logs.json` and `trade_decisions.json`) to GitHub. Logs are also committed upon job completion to ensure all decisions are tracked.
- **Error Handling**: Alerts are sent via email in case of API errors, connection issues, or trading exceptions.

### Automated Trading Workflow
- **Scheduled Execution**: The bot automatically starts at **9:25 AM EST** and runs until **3:50 PM EST**. 
- **Manual Trigger**: The workflow can also be manually triggered from the GitHub Actions tab.

## Getting Started

### Prerequisites

- Python 3.12.6 (or compatible version)
- An account with Alpaca for Paper Trading ([Sign up here](https://alpaca.markets/))
- An OpenAI API key ([Get your API key](https://openai.com))
- A News API key from Finnhub or similar service ([Get API key](https://finnhub.io))

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/TradingBot.git
    cd TradingBot
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Add your API keys to the `bot.py` file:
    - Alpaca API key
    - Alpaca API secret
    - OpenAI API key
    - News API key

### Running the Bot Locally

You can run the trading bot manually using the following command:
```bash
python bot.py
```

### GitHub Actions Workflow

The bot is designed to run automatically via GitHub Actions. It includes the following steps:

1. **Python Setup**: Installs the appropriate version of Python and required packages (including `ta-lib`).
2. **Bot Execution**: Runs the bot during market hours.
3. **Log Management**: Commits logs of trading decisions and GPT-3 responses to the repository.

## Workflow Configuration

The bot runs automatically at **9:25 AM EST** and stops at **3:50 PM EST**, using the following cron schedule:

```yaml
on:
  schedule:
    - cron: '25 13 * * 1-5'  # Daylight Savings
    - cron: '25 14 * * 1-5'  # Standard Time
  workflow_dispatch:
```

### Log Files

- **chat_gpt_logs.json**: Logs all OpenAI chat responses regarding buy/sell decisions.
- **trade_decisions.json**: Logs all buy/sell/hold decisions made by the bot.

These logs are periodically committed to GitHub, ensuring every decision is tracked.

### Error Handling

If the bot encounters any errors, such as API call failures or stream disconnections, it will:
- Send an email alert to the designated email.
- Retry certain failed actions like API calls.

## Liquidation

- At **3:50 PM EST**, the bot liquidates all positions and halts further trades.
- There is a 10-minute waiting period before the bot shuts down, ensuring that no further trades are made after liquidation.

## Contributing

We welcome contributions! Here's how you can get started:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

