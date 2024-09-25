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
import smtplib
from email.mime.text import MIMEText
import asyncio
import pytz
from datetime import datetime, time as dtime
# Function to query GPT with trading log reference
from transformers import pipeline

logging.basicConfig(level=logging.INFO)

final_position_size = 0

#  Email setup for error notifications
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USER = ''  # Replace with your email
EMAIL_PASS = ''  # Replace with your email password
TO_EMAIL = ''  # Replace with destination email for alerts

# Send email notification for errors
def send_email(subject, message):
    try:
        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = TO_EMAIL

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, TO_EMAIL, msg.as_string())
            print("Error notification sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

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
TRAILING_STOP_PERCENT = 0.0001  # Trailing stop of .1 %

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
                send_email("GitHub Commit Error", f"Failed to commit logs: {e}")

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

# Error handling function for Alpaca API calls
def api_call_with_retry(func, *args, **kwargs):
    retries = 1
    for i in range(retries):
        try:
            return func(*args, **kwargs)  # Remove timeout
        except Exception as e:
            logging.error(f"API call failed: {e}")
            send_email(f"API Error on Attempt {i+1}", str(e))
            if i < retries - 1:
                time.sleep(3)
            else:
                raise

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
    
    # Finnhub API URL for fetching company news
    url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2023-01-01&to=2023-12-31&token={NEWS_API_KEY}'
    
    response = requests.get(url)
    news_data = response.json()

    if response.status_code == 200:
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

# Load GPT-J model from Hugging Face
generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')

def query_huggingface_for_decision(ticker, rsi_value, sentiment_score, pe_ratio, var):
    print(f"Querying Hugging Face GPT-J for decision on {ticker}")
    
    # Prepare the prompt with stock data
    prompt = f"""
    Here is the data for the stock {ticker}:
    - RSI value: {rsi_value}
    - Sentiment score: {sentiment_score}
    - P/E ratio: {pe_ratio}
    - Value at Risk (VaR): {var}

    Based on this data, should we buy or sell this stock? Only reply with a single word 'BUY', 'SELL' or 'NO ACTION'.
    """

    try:
        # Generate response from Hugging Face model with temperature
        response = generator(prompt, max_length=50, do_sample=True, temperature=0.7)  # Added temperature control
        decision = response[0]['generated_text'].strip().split()[-1]  # Extract the last word for decision

        print(f"Hugging Face GPT-J decision for {ticker}: {decision}")

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
        print(f"Error using Hugging Face GPT-J: {e}")

        # Return None if Hugging Face model call fails
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
        #final_position_size = int(min(position_size, max_shares_based_on_allocation, available_shares))

        final_position_size = round(position_size)
        
        print(f"Final position size for {ticker}: {final_position_size} shares")
        return final_position_size
    else:
        print(f"VaR is zero for {ticker}, skipping trade.")
        return 0

# Function to set a trailing stop loss for a trade
def set_trailing_stop(ticker, position_size):
    try:

        time.sleep(3)
        check_latest_order_filled(ticker)

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
                trail_percent=0.01  # Adjust this to the trailing stop percent you prefer
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
    decision = query_huggingface_for_decision(ticker, data['RSI'].iloc[-1], sentiment_score, pe_ratio, var)
    return decision

def trade_stock(ticker, decision, portfolio_value):
    print(f"Placing trade for {ticker}")
    
    # Check if decision is None and log the error
    if decision is None:
        print(f"Skipping trade for {ticker} due to missing decision from OpenAI.")
        log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "NO_ACTION", "reason": "Missing decision from OpenAI"})
        return

    # Calculate position size based on portfolio value
    position_size = calculate_position_size(ticker, portfolio_value)

    if position_size > 0 and has_sufficient_buying_power(ticker, position_size):
        if 'buy' in decision.lower():

            check_latest_order_filled(ticker)

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

            check_latest_order_filled(ticker)

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
        print(f"Position size for {ticker} is zero or insufficient buying power, no trade made.")
        log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": "NO_ACTION", "reason": "Zero position size or insufficient buying power"})


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


# WebSocket price handler to evaluate stocks in real-time
async def on_bar(data):
    try:
        ticker = data.symbol
        account = api_call_with_retry(api.get_account)
        portfolio_value = float(account.equity)
        logging.info(f"Received price update for {ticker}. Portfolio value: ${portfolio_value}")
        
        decision = evaluate_stock(ticker, portfolio_value)
        trade_stock(ticker, decision, portfolio_value)
    except Exception as e:
        logging.error(f"Error during stream handling for {ticker}: {e}")
        send_email(f"Error during stream handling for {ticker}", str(e))


# Set up WebSocket subscription for S&P 500 tickers
def subscribe_to_tickers():
    print("Subscribing to tickers: AAPL, MSFT, GOOGL, AMZN, TSLA")
    sp500_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'GME', 'AMC', 'NVDA', 'AMD', 'META', 'NFLX']  # Example tickers
    for ticker in sp500_tickers:
        stream.subscribe_bars(on_bar, ticker)

# Function to calculate performance metrics
def calculate_performance_metrics(portfolio_value):
    pnl = portfolio_value - INITIAL_PORTFOLIO_VALUE
    return {"PnL": pnl}

# Function to check if there is sufficient buying power
def has_sufficient_buying_power(ticker, position_size):
    account = api_call_with_retry(api.get_account)
    buying_power = float(account.buying_power)
    last_trade = api_call_with_retry(api.get_latest_trade, ticker)
    stock_price = last_trade.price
    required_amount = stock_price * position_size

    if buying_power >= required_amount:
        logging.info(f"Sufficient buying power to buy {position_size} shares of {ticker}")
        return True
    else:
        logging.error(f"Insufficient buying power: Need ${required_amount}, but only have ${buying_power}")
        #send_email("Insufficient Buying Power", f"Need ${required_amount}, but only have ${buying_power}")
        return False
#liquidation commmented out
"""
# Define Eastern Time Zone
eastern = pytz.timezone('US/Eastern')

# Function to liquidate all positions
def liquidate_positions():
    print("Liquidating all positions...")
    try:
        api.close_all_positions()
        log_decision({"time": datetime.now().isoformat(), "action": "LIQUIDATE", "reason": "End of day liquidation at 3:50 PM EST"})
        print("All positions liquidated.")
    except Exception as e:
        print(f"Error during liquidation: {e}")
        send_email("Liquidation Error", str(e))

# Function to check if it's 3:50 PM EST and liquidate all positions
def check_time_for_liquidation():
    current_time = datetime.now(pytz.utc).astimezone(eastern).time()  # Get current time in EST
    liquidation_time = dtime(15, 50)  # 3:50 PM Eastern Time
    if current_time >= liquidation_time:
        liquidate_positions()
        # Wait for 10 minutes (600 seconds) after liquidation
        print("Waiting for 10 minutes before shutting down...")
        time.sleep(600)  # 10-minute wait
        # End the program to prevent further trades
        print("Shutting down the bot after end-of-day liquidation.")
        sys.exit(0)  # Cleanly exit the program

# Add this function call to your main loop or as part of your scheduler to run continuously.
def run_liquidation_scheduler():
    while True:
        check_time_for_liquidation()
        time.sleep(60)  # Check every 60 secs

# Start the liquidation scheduler in a background thread
liquidation_thread = threading.Thread(target=run_liquidation_scheduler, daemon=True)
liquidation_thread.start()
"""

# Automatically restart stream on connection errors
async def start_stream():
    while True:
        try:
            logging.info("Starting WebSocket stream...")
            await stream._run_forever()  # Use the proper async call here
        except Exception as e:
            logging.error(f"Stream error: {e}")
            send_email("Stream Error", str(e))
            await asyncio.sleep(2)  # Use asyncio.sleep instead of time.sleep

# Main function to run the trading bot
async def run_trading_bot():
    logging.info("Starting trading bot...")
    if check_portfolio() and check_drawdown():
        subscribe_to_tickers()
        await start_stream()  # Use await here


if __name__ == '__main__':
    try:
        # asyncio.run is the right way to launch it outside an event loop
        asyncio.run(run_trading_bot())
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        send_email("Bot Crashed", str(e))
        sys.exit(1)
