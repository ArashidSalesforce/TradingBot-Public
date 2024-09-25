import openai
import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
import json
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from alpaca_trade_api.stream import Stream
import math
import logging
import os
import time
import signal
import subprocess
import sys
import threading

logging.basicConfig(level=logging.INFO)

# Use logging instead of print
logging.info("Starting trading bot...")


# Alpaca API credentials
API_KEY = ''
API_SECRET = ''
BASE_URL = 'https://paper-api.alpaca.markets'
WS_URL = 'wss://stream.data.alpaca.markets'

# OpenAI API key for g-0P2AAnxSV-trading-bot
openai.api_key = ''

# News API credentials
NEWS_API_KEY = ''

# Initialize Alpaca API and WebSocket Stream
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
stream = Stream(API_KEY, API_SECRET, WS_URL, data_feed='iex')

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Risk parameters
INITIAL_PORTFOLIO_VALUE = 26000
MIN_PORTFOLIO_VALUE = 25100
MAX_RISK_PER_TRADE = 0.01  # Risk 1% of the portfolio value per trade
MAX_DRAW_DOWN_LIMIT = 0.08  # Stop trading if drawdown exceeds 8%
MAX_STOCK_ALLOCATION = 0.20  # Maximum 20% of portfolio in one stock
TRAILING_STOP_PERCENT = 0.05  # Trailing stop of 5%

# JSON log file
LOG_FILE = 'trading_log.json'

last_commit_time = time.time()
lock = threading.Lock()

def commit_logs():
    global last_commit_time
    current_time = time.time()
    with lock:
        if current_time - last_commit_time > 1800:  # Every 30 minutes
            try:
                subprocess.run(["git", "add", "chat_gpt_logs.json", "trade_decisions.json"], check=True)
                subprocess.run(["git", "commit", "-m", "Update chat_gpt_logs and trade_decisions"], check=True)
                subprocess.run(["git", "push"], check=True)
                print("Logs committed and pushed.")
                last_commit_time = current_time
            except subprocess.CalledProcessError as e:
                print(f"Error committing logs: {e}")

def commit_logs_periodically():
    while True:
        commit_logs()
        time.sleep(1800)  # Commit every 30 minutes

# Start background thread for periodic commit
log_commit_thread = threading.Thread(target=commit_logs_periodically, daemon=True)
log_commit_thread.start()


def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Shutting down gracefully...')
    # Perform any necessary cleanup here
    stream.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Function to log decisions
def log_decision(log_entry):
    log_file = "trade_decisions.json"

    # Check if the log file exists
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([], f)  # Initialize with an empty list

    # Read the current logs
    with open(log_file, 'r') as f:
        logs = json.load(f)
        if not isinstance(logs, list):
            logs = []  # Ensure logs is a list

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
    var = np.percentile(returns, (1 - confidence_level) * 100)
    print(f"VaR for {ticker}: {var}")
    return var * math.sqrt(10)  # 10-day VaR

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
    url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    news_data = response.json()
    if news_data.get('status') == 'ok':
        total_score = 0
        articles = news_data['articles']
        for article in articles:
            sentiment_score = analyzer.polarity_scores(article['title'])
            total_score += sentiment_score['compound']
        avg_sentiment = total_score / len(articles) if articles else 0
        print(f"Average sentiment for {ticker}: {avg_sentiment}")
        return avg_sentiment
    return 0

# Function to load and summarize the trading log
def load_trading_log():
    try:
        with open(LOG_FILE, 'r') as f:
            logs = [json.loads(line) for line in f.readlines()]
            return logs
    except FileNotFoundError:
        print("Trading log not found, continuing without it.")
        return []

def summarize_trading_log(logs, ticker):
    summary = f"Summary of previous trades for {ticker}:\n"
    for log in logs:
        if log['ticker'] == ticker:
            qty = log.get('qty', 'unknown')  # Handle missing 'qty' key
            summary += f"On {log['time']}, action: {log['action']} for {qty} shares.\n"
    return summary


# Function to query GPT with trading log reference
def query_openai_for_decision(ticker, rsi_value, sentiment_score, pe_ratio, var):
    print(f"Querying OpenAI for decision on {ticker}")

    # Create the prompt with only the stock data
    messages = [
        {
            "role": "system",
            "content": "You are a trading bot with extensive knowledge of financial markets, risk management, and factor models."
        },
        {
            "role": "user",
            "content": f"""
            Here is the data for the stock {ticker}:
            - RSI value: {rsi_value}
            - Sentiment score: {sentiment_score}
            - P/E ratio: {pe_ratio}
            - Value at Risk (VaR): {var}

            Based on this data, should we buy or sell this stock? Provide reasoning and strategy.
            """
        }
    ]

    try:
        # Request decision from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=4000,
            temperature=0.7,
        )
        # Extract decision from GPT response
        decision = response['choices'][0]['message']['content'].strip()
        print(f"OpenAI decision for {ticker}: {decision}")

        # Log the interaction into a JSON file
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "rsi_value": rsi_value,
            "sentiment_score": sentiment_score,
            "pe_ratio": pe_ratio,
            "var": var,
            "decision": decision
        }

        log_to_json(log_data)

        return decision

    except Exception as e:
        print(f"Error using OpenAI: {e}")
        # Return None if OpenAI call fails
        return None


# Function to append log to JSON
def log_to_json(log_data):
    log_file = "chat_gpt_logs.json"

    # Check if the log file exists, if not create it with an empty list
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([], f)  # Initialize the file with an empty list

    # Read the current logs, with a fallback in case the file is corrupted or empty
    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
            if not isinstance(logs, list):
                logs = []  # Ensure logs is a list if somehow it's not
    except (json.JSONDecodeError, FileNotFoundError):
        logs = []  # In case the file is corrupted or empty

    # Append the new log data
    logs.append(log_data)

    # Write the updated logs back to the file
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)

    print(f"Logged chat interaction for {log_data.get('ticker', 'Unknown Ticker')}.")


# Function to calculate position size based on VaR and portfolio risk management
def calculate_position_size(ticker, portfolio_value, max_risk_per_trade=MAX_RISK_PER_TRADE):
    print(f"Calculating position size for {ticker}")
    var = calculate_var(ticker)
    
    # Risk per trade (e.g., 1% of portfolio)
    risk_per_trade = portfolio_value * max_risk_per_trade
    last_trade = api.get_latest_trade(ticker)
    stock_price = last_trade.price

    if var != 0:
        position_size = risk_per_trade / (stock_price * abs(var))
        max_shares_based_on_allocation = (portfolio_value * MAX_STOCK_ALLOCATION) // stock_price
        final_position_size = int(min(position_size, max_shares_based_on_allocation))
        print(f"Position size for {ticker}: {final_position_size} shares")
        return final_position_size
    else:
        print(f"VaR is zero for {ticker}, skipping trade.")
        return 0

# Function to set a trailing stop loss for a trade
def set_trailing_stop(ticker, position_size):
    try:
        # Fetch the current position for the stock
        position = api.get_position(ticker)
        
        # Ensure that we are only setting a trailing stop for a long position
        if position.side == 'long':
            print(f"Setting trailing stop for {ticker} with {position_size} shares.")
            
            # Define the stop loss parameters (modify according to your strategy)
            api.submit_order(
                symbol=ticker,
                qty=position_size,
                side='sell',
                type='trailing_stop',
                time_in_force='gtc',  # Good-till-cancelled
                trail_percent=2.0  # Adjust this to the trailing stop percent you prefer
            )
        else:
            print(f"Cannot set trailing stop for a short or non-existent position on {ticker}.")
    except Exception as e:
        print(f"Error while setting trailing stop for {ticker}: {e}")


# Alpha Factor Model: Combine RSI (momentum), P/E ratio (value), sentiment, and use g-0P2AAnxSV-trading-bot for final decision
def evaluate_stock(ticker, portfolio_value):
    print(f"Evaluating stock: {ticker}")
    data = yf.download(ticker, period='1mo', interval='1d')
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    sentiment_score = get_news_sentiment(ticker)
    stock_info = yf.Ticker(ticker).info
    pe_ratio = stock_info.get('trailingPE', None)
    var = calculate_var(ticker)
    decision = query_openai_for_decision(ticker, data['RSI'].iloc[-1], sentiment_score, pe_ratio, var)
    return decision

# Function to place trades based on decision
def trade_stock(ticker, decision, portfolio_value):
    print(f"Placing trade for {ticker}")
    
    # Check if decision is None and log the error
    if decision is None:
        print(f"Skipping trade for {ticker} due to missing decision from OpenAI.")
        log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "NO_ACTION", "reason": "Missing decision from OpenAI"})
        return

    # Calculate position size based on portfolio value
    position_size = calculate_position_size(ticker, portfolio_value)

    if position_size > 0:
        if 'buy' in decision.lower():
            # Submit a buy order
            api.submit_order(
                symbol=ticker,
                qty=position_size,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            print(f"Buying {position_size} shares of {ticker}")
            
            # Set a trailing stop loss for the purchased shares
            set_trailing_stop(ticker, position_size)
            
            # Log the decision
            log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "BUY", "qty": position_size})
        
        elif 'sell' in decision.lower():
            # Submit a sell order
            api.submit_order(
                symbol=ticker,
                qty=position_size,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            print(f"Selling {position_size} shares of {ticker}")
            
            # Log the decision
            log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "SELL", "qty": position_size})
        
        else:
            print(f"No clear buy/sell action for {ticker}, decision was: {decision}")
            log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "NO_ACTION", "reason": f"Unclear decision: {decision}"})
    else:
        print(f"Position size for {ticker} is zero, no trade made.")
        log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "NO_ACTION", "reason": "Zero position size"})

# WebSocket price handler to evaluate stocks in real-time
async def on_bar(data):
    ticker = data.symbol
    account = api.get_account()
    portfolio_value = float(account.equity)
    print(f"Received price update for {ticker}. Portfolio value: ${portfolio_value}")
    
    decision = evaluate_stock(ticker, portfolio_value)
    trade_stock(ticker, decision, portfolio_value)

# Set up WebSocket subscription for S&P 500 tickers
def subscribe_to_tickers():
    print("Subscribing to tickers: AAPL, MSFT, GOOGL, AMZN, TSLA")
    sp500_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Example tickers
    for ticker in sp500_tickers:
        stream.subscribe_bars(on_bar, ticker)

# Main function to run the trading bot
def run_trading_bot():
    print("Starting trading bot...")
    if check_portfolio() and check_drawdown():
        subscribe_to_tickers()
        stream.run()

if __name__ == '__main__':
    run_trading_bot()
