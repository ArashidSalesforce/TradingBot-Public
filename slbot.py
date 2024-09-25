import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
import json
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from alpaca_trade_api.stream import Stream
import math
import os
import time
import smtplib
from email.mime.text import MIMEText
import asyncio

# Email setup for error notifications
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USER = 'your_email@example.com'  # Replace with your email
EMAIL_PASS = 'your_email_password'     # Replace with your email password
TO_EMAIL = 'destination_email@example.com'  # Replace with destination email for alerts

print("Starting trading bot...")

# Alpaca API credentials
API_KEY = ''        # Replace with your Alpaca API Key
API_SECRET = ''  # Replace with your Alpaca Secret Key
BASE_URL = 'https://paper-api.alpaca.markets'
WS_URL = 'wss://stream.data.alpaca.markets'

# News API credentials
NEWS_API_KEY = ''     # Replace with your Finnhub API Key

# Initialize Alpaca API and WebSocket Stream
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
stream = Stream(API_KEY, API_SECRET, WS_URL, data_feed='sip')

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Risk parameters
INITIAL_PORTFOLIO_VALUE = 26000
MIN_PORTFOLIO_VALUE = 25100
MAX_RISK_PER_TRADE = 0.01  # Risk 1% of the portfolio value per trade
MAX_DRAW_DOWN_LIMIT = 0.08  # Stop trading if drawdown exceeds 8%
MAX_STOCK_ALLOCATION = 0.20  # Maximum 20% of portfolio in one stock
TRAILING_STOP_PERCENT = 0.0009  # Trailing stop loss percentage (0.1%)

# JSON log files
LOG_FILE = 'trading_log.json'
HIGHEST_PRICE_FILE = 'highest_price.json'

# Dictionary to track the highest price since purchase for each position
position_highest_price = {}

# Load the highest price data from file
def load_highest_price_data():
    global position_highest_price
    if os.path.exists(HIGHEST_PRICE_FILE):
        with open(HIGHEST_PRICE_FILE, 'r') as f:
            position_highest_price = json.load(f)
            # Convert string keys to float
            position_highest_price = {symbol: float(price) for symbol, price in position_highest_price.items()}
        print("Loaded highest price data.")
    else:
        position_highest_price = {}
        print("No highest price data found. Starting fresh.")

# Save the highest price data to file
def save_highest_price_data():
    with open(HIGHEST_PRICE_FILE, 'w') as f:
        json.dump(position_highest_price, f)
    print("Saved highest price data.")

# Function to log decisions
def log_decision(log_entry):
    log_file = "trade_decisions.json"

    # Check if the log file exists
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([], f)  # Initialize with an empty list

    # Read the current logs
    with open(log_file, 'r') as f:
        try:
            logs = json.load(f)
            if not isinstance(logs, list):
                logs = []  # Ensure logs is a list
        except json.JSONDecodeError:
            logs = []

    # Append the new log entry to the list
    logs.append(log_entry)

    # Write the updated logs back to the file
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)

    print(f"Logged decision for {log_entry['ticker']} action: {log_entry['action']}")

# Function to check portfolio value and liquidate if below threshold
def check_portfolio():
    account = api.get_account()
    portfolio_value = float(account.equity)
    print(f"Checking portfolio value: ${portfolio_value}")
    if portfolio_value < MIN_PORTFOLIO_VALUE:
        print(f"Portfolio below minimum value! Current: ${portfolio_value}. Liquidating positions.")
        api.close_all_positions()
        log_decision({"time": datetime.now().isoformat(), "action": "LIQUIDATE", "reason": "Portfolio value below threshold"})
        return False
    return True

# Function to calculate Value at Risk (VaR)
def calculate_var(ticker, confidence_level=0.95):
    print(f"Calculating VaR for {ticker}")
    data = yf.download(ticker, period='3mo', interval='1d')
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    if not returns.empty:
        var = np.percentile(returns, (1 - confidence_level) * 100)
        print(f"VaR for {ticker}: {var}")
        return var * math.sqrt(10)  # 10-day VaR
    else:
        print(f"Insufficient data to calculate VaR for {ticker}")
        return 0

# Function to calculate drawdown
def calculate_drawdown(portfolio_value):
    print(f"Calculating drawdown for portfolio value: ${portfolio_value}")
    historical_data = api.get_portfolio_history()
    peak = max(historical_data.equity)
    drawdown = (peak - portfolio_value) / peak
    print(f"Drawdown: {drawdown * 100:.2f}%")
    return drawdown

# Function to check for drawdown and stop trading if necessary
def check_drawdown():
    portfolio_value = float(api.get_account().equity)
    drawdown = calculate_drawdown(portfolio_value)
    if drawdown > MAX_DRAW_DOWN_LIMIT:
        print(f"Max drawdown exceeded! Stopping trading. Drawdown: {drawdown * 100:.2f}%")
        api.close_all_positions()
        log_decision({"time": datetime.now().isoformat(), "action": "STOP_TRADING", "reason": "Max drawdown exceeded"})
        return False
    return True

# Function to get news sentiment using VADER Sentiment Analysis
def get_news_sentiment(ticker):
    print(f"Fetching news sentiment for {ticker}")

    # Finnhub API URL for fetching company news
    url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2023-01-01&to=2023-12-31&token={NEWS_API_KEY}'

    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        total_score = 0
        articles = news_data
        for article in articles:
            if 'headline' in article:
                sentiment_score = analyzer.polarity_scores(article['headline'])
                total_score += sentiment_score['compound']
        avg_sentiment = total_score / len(articles) if articles else 0
        print(f"Average sentiment for {ticker}: {avg_sentiment}")
        return avg_sentiment
    else:
        print(f"Error fetching news for {ticker}, Status Code: {response.status_code}")
        return 0

# Error handling function for Alpaca API calls
def api_call_with_retry(func, *args, **kwargs):
    retries = 1
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"API call failed: {e}")
            if i < retries - 1:
                time.sleep(3)
            else:
                raise

# Function to calculate position size based on VaR and portfolio risk management
def calculate_position_size(ticker, portfolio_value, max_risk_per_trade=MAX_RISK_PER_TRADE):
    print(f"Calculating position size for {ticker}")

    # Calculate VaR for risk management
    var = calculate_var(ticker)

    # Risk per trade (e.g., 1% of portfolio)
    risk_per_trade = portfolio_value * max_risk_per_trade

    # Get the latest stock price
    last_trade = api_call_with_retry(api.get_latest_trade, ticker)
    stock_price = last_trade.price

    # Ensure we have valid VaR, else skip the trade
    if var != 0:
        # Calculate position size based on risk
        position_size = risk_per_trade / (stock_price * abs(var))

        # Ensure the position does not exceed maximum allocation
        max_shares_based_on_allocation = (portfolio_value * MAX_STOCK_ALLOCATION) // stock_price

        # Fetch available buying power
        account = api_call_with_retry(api.get_account)
        buying_power = float(account.buying_power)

        # Calculate maximum possible shares based on buying power
        available_shares = buying_power // stock_price

        # Final position size should be the minimum of calculated size, max allocation, and available funds
        final_position_size = int(min(position_size, max_shares_based_on_allocation, available_shares))

        print(f"Final position size for {ticker}: {final_position_size} shares")
        return final_position_size
    else:
        print(f"VaR is zero for {ticker}, skipping trade.")
        return 0

# Function to check if the latest order for a ticker has been filled
def check_latest_order_filled(ticker):
    print(f"Checking latest order status for {ticker}")

    # Retrieve the most recent order for the given stock
    try:
        orders = api.list_orders(
            status='all',  # Retrieve all orders, including pending and completed
            symbols=[ticker],  # Filter by the stock symbol
            limit=1,  # Only get the most recent order
            direction='desc'  # Sort orders by most recent first
        )

        # If there are no orders, return True (proceed with new order)
        if not orders:
            print(f"No previous orders found for {ticker}. Proceeding with new order.")
            return True

        # Check the status of the latest order
        latest_order = orders[0]
        order_status = latest_order.status  # Status could be 'filled', 'partially_filled', 'new', 'cancelled', etc.

        # Log the order status for debugging
        print(f"Latest order status for {ticker}: {order_status}")

        # Return True if the latest order is filled or cancelled, meaning we can place a new order
        if order_status in ['filled', 'cancelled']:
            return True
        else:
            print(f"Cannot place a new order. Latest order for {ticker} is still active with status: {order_status}")
            return False

    except Exception as e:
        print(f"Error checking latest order for {ticker}: {e}")
        return False

# Function to evaluate stock based on metrics and make a decision
def evaluate_stock(ticker, portfolio_value):
    print(f"Evaluating stock: {ticker}")
    data = yf.download(ticker, period='1mo', interval='1d')
    if data.empty:
        print(f"No data for {ticker}, skipping evaluation.")
        return "NO_ACTION"

    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['EMA_20'] = EMAIndicator(data['Close'], window=20).ema_indicator()
    data['EMA_50'] = EMAIndicator(data['Close'], window=50).ema_indicator()

    # Get the latest values
    rsi_value = data['RSI'].iloc[-1]
    ema_20 = data['EMA_20'].iloc[-1]
    ema_50 = data['EMA_50'].iloc[-1]
    close_price = data['Close'].iloc[-1]

    sentiment_score = get_news_sentiment(ticker)
    var = calculate_var(ticker)

    # Decision logic based on metrics
    decision = "NO_ACTION"

    # Buy if RSI > 50, EMA_20 > EMA_50 (uptrend), and sentiment is positive
    if rsi_value > 50 and ema_20 > ema_50 and sentiment_score > 0:
        decision = "BUY"
    # Sell if RSI < 50, EMA_20 < EMA_50 (downtrend), and sentiment is negative
    elif rsi_value < 50 and ema_20 < ema_50 and sentiment_score < 0:
        decision = "SELL"

    print(f"Decision for {ticker}: {decision}")
    return decision

def trade_stock(ticker, decision, portfolio_value):
    print(f"Placing trade for {ticker}")

    # Check if decision is None and log the error
    if decision is None or decision == "NO_ACTION":
        print(f"Skipping trade for {ticker} due to decision: {decision}.")
        log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "NO_ACTION", "reason": f"Decision: {decision}"})
        return

    # Calculate position size based on portfolio value
    position_size = calculate_position_size(ticker, portfolio_value)

    if position_size > 0 and has_sufficient_buying_power(ticker, position_size):
        if decision == "BUY":

            if not check_latest_order_filled(ticker):
                return

            # Submit a buy order
            api.submit_order(
                symbol=ticker,
                qty=position_size,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            print(f"Buying {position_size} shares of {ticker}")

            # Initialize highest price to purchase price
            last_trade = api.get_latest_trade(ticker)
            position_highest_price[ticker] = last_trade.price
            save_highest_price_data()

            # Log the decision
            log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "BUY", "qty": position_size})

        elif decision == "SELL":

            if not check_latest_order_filled(ticker):
                return

            # Submit a sell order
            api.submit_order(
                symbol=ticker,
                qty=position_size,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            print(f"Selling {position_size} shares of {ticker}")

            # Remove the ticker from highest price tracking
            if ticker in position_highest_price:
                del position_highest_price[ticker]
                save_highest_price_data()

            # Log the decision
            log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "SELL", "qty": position_size})

        else:
            print(f"No clear buy/sell action for {ticker}, decision was: {decision}")
            log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "NO_ACTION", "reason": f"Unclear decision: {decision}"})
    else:
        print(f"Position size for {ticker} is zero or insufficient buying power, no trade made.")
        log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "NO_ACTION", "reason": "Zero position size or insufficient buying power"})

# Function to check if there is sufficient buying power
def has_sufficient_buying_power(ticker, position_size):
    account = api_call_with_retry(api.get_account)
    buying_power = float(account.buying_power)
    last_trade = api_call_with_retry(api.get_latest_trade, ticker)
    stock_price = last_trade.price
    required_amount = stock_price * position_size

    if buying_power >= required_amount:
        print(f"Sufficient buying power to buy {position_size} shares of {ticker}")
        return True
    else:
        print(f"Insufficient buying power: Need ${required_amount}, but only have ${buying_power}")
        return False

# WebSocket price handler to evaluate stocks in real-time
async def on_bar(data):
    try:
        ticker = data.symbol
        # Get the current account equity
        account = api_call_with_retry(api.get_account)
        portfolio_value = float(account.equity)
        print(f"Received price update for {ticker}. Portfolio value: ${portfolio_value}")

        # Evaluate the stock and make a decision
        decision = evaluate_stock(ticker, portfolio_value)
        trade_stock(ticker, decision, portfolio_value)

        # Now check positions and sell if necessary
        # Fetch current positions
        positions = api.list_positions()
        for position in positions:
            symbol = position.symbol
            qty = abs(int(float(position.qty)))
            current_price = float(position.current_price)

            # Initialize highest price if not present
            if symbol not in position_highest_price:
                position_highest_price[symbol] = current_price
                save_highest_price_data()

            # Update the highest price
            if current_price > position_highest_price[symbol]:
                position_highest_price[symbol] = current_price
                save_highest_price_data()
                print(f"Updated highest price for {symbol}: {position_highest_price[symbol]}")

            # Calculate the trailing stop price
            trailing_stop_price = position_highest_price[symbol] * (1 - TRAILING_STOP_PERCENT)

            # Check if the current price has dipped below the trailing stop price
            if current_price < trailing_stop_price:
                # Sell the position
                print(f"Current price of {symbol} has dipped below trailing stop loss price. Selling position.")
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell' if position.side == 'long' else 'buy',
                    type='market',
                    time_in_force='gtc'
                )
                log_decision({
                    "time": datetime.now().isoformat(),
                    "ticker": symbol,
                    "action": "SELL",
                    "qty": qty,
                    "reason": "Price dipped below trailing stop loss price"
                })
                # Remove from highest price tracking
                if symbol in position_highest_price:
                    del position_highest_price[symbol]
                    save_highest_price_data()
    except Exception as e:
        print(f"Error during stream handling for {data.symbol}: {e}")

# Set up WebSocket subscription for tickers
def subscribe_to_tickers():
    print("Subscribing to tickers: AAPL, MSFT, GOOGL, AMZN, TSLA, GME, AMC")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'GME', 'AMC', 'HPQ']  # Example tickers
    print(tickers)
    for ticker in tickers:
        stream.subscribe_bars(on_bar, ticker)

# Automatically restart stream on connection errors
async def start_stream():
    while True:
        try:
            print("Starting WebSocket stream...")
            await stream._run_forever()
        except Exception as e:
            print(f"Stream error: {e}")
            await asyncio.sleep(2)

# Main function to run the trading bot
async def run_trading_bot():
    print("Starting trading bot...")
    load_highest_price_data()
    if check_portfolio() and check_drawdown():
        subscribe_to_tickers()
        await start_stream()

if __name__ == '__main__':
    try:
        asyncio.run(run_trading_bot())
    except Exception as e:
        print(f"Unhandled exception: {e}")
