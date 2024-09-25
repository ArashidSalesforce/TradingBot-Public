import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
import json
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
import math
import logging
import os
import time
import threading
import smtplib
from email.mime.text import MIMEText
import asyncio
import pytz
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
import joblib
import xgboost as xgb

logging.basicConfig(level=logging.INFO)

# Email setup for error notifications
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USER = ''  # Replace with your email
EMAIL_PASS = ''  # Replace with your email password
TO_EMAIL = ''  # Replace with destination email for alerts

# Alpaca API credentials
API_KEY = ''
API_SECRET = ''
BASE_URL = 'https://paper-api.alpaca.markets'
WS_URL = 'wss://stream.data.alpaca.markets'

# News API credentials (e.g., Finnhub)
NEWS_API_KEY = ''

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Risk parameters
INITIAL_PORTFOLIO_VALUE = 100000
MAX_RISK_PER_TRADE = 0.005
MAX_DRAW_DOWN_LIMIT = 0.05
MAX_STOCK_ALLOCATION = 0.10
TRAILING_STOP_MULTIPLIER = 3

# Transaction cost assumptions
TRANSACTION_COST = 0.0005  # 0.05% per trade

# JSON log file
LOG_FILE = 'trading_log.json'

# Define Eastern Time Zone
eastern = pytz.timezone('US/Eastern')

# List of tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'XOM', 'JNJ', 'WMT']

# Logging setup
logger = logging.getLogger('TradingBot')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('trading_bot.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Machine Learning Model (will be set after hyperparameter tuning)
best_model = None

# Function to send email notifications for errors
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
            print("Notification sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to log decisions
def log_decision(log_entry):
    log_file = "trade_decisions.json"

    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([], f)

    with open(log_file, 'r') as f:
        try:
            logs = json.load(f)
            if not isinstance(logs, list):
                logs = []
        except json.JSONDecodeError:
            logs = []

    logs.append(log_entry)

    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)

    print(f"Logged decision for {log_entry['ticker']} action: {log_entry['action']}")

# Feature Engineering Function
def prepare_ml_data(ticker):
    data = yf.download(ticker, period='2y', interval='1d')
    data.dropna(inplace=True)

    # Technical Indicators
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    macd_indicator = MACD(data['Close'])
    data['MACD'] = macd_indicator.macd()
    data['MACD_Signal'] = macd_indicator.macd_signal()
    data['MACD_Hist'] = macd_indicator.macd_diff()
    data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()

    # Fill NaN values
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)

    # Target Variable
    data['Future_Return'] = data['Close'].shift(-1) / data['Close'] - 1
    data['Target'] = (data['Future_Return'] > TRANSACTION_COST).astype(int)

    # Features and Labels
    features = data[['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR']]
    labels = data['Target']

    return features[:-1], labels[:-1]  # Exclude the last row due to NaN in 'Future_Return'

# Hyperparameter Tuning Function
def hyperparameter_tuning(X, y):
    logger.info("Starting hyperparameter tuning...")

    classifiers = {
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    param_grids = {
        'RandomForest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'GradientBoosting': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    }

    best_score = 0
    best_model = None
    best_params = None

    tscv = TimeSeriesSplit(n_splits=5)

    for clf_name in classifiers:
        logger.info(f"Tuning {clf_name}...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifiers[clf_name])
        ])

        param_grid = param_grids[clf_name]

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=10,
            scoring='accuracy',
            cv=tscv,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X, y)
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            best_params = search.best_params_

    logger.info(f"Best Model: {best_model.named_steps['classifier'].__class__.__name__}")
    logger.info(f"Best Score: {best_score}")
    logger.info(f"Best Parameters: {best_params}")

    return best_model

# Function to Train ML Model and Evaluate
def train_and_evaluate_model():
    logger.info("Training and evaluating the machine learning model...")
    all_features = pd.DataFrame()
    all_labels = pd.Series(dtype=int)

    for ticker in tickers:
        features, labels = prepare_ml_data(ticker)
        all_features = all_features.append(features)
        all_labels = all_labels.append(labels)

    # Handle missing values
    all_features.fillna(method='ffill', inplace=True)
    all_features.fillna(method='bfill', inplace=True)

    # Hyperparameter Tuning
    best_model = hyperparameter_tuning(all_features, all_labels)

    # Model Evaluation Metrics
    tscv = TimeSeriesSplit(n_splits=5)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []

    for train_index, test_index in tscv.split(all_features):
        X_train, X_test = all_features.iloc[train_index], all_features.iloc[test_index]
        y_train, y_test = all_labels.iloc[train_index], all_labels.iloc[test_index]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        roc_auc_scores.append(roc_auc_score(y_test, y_proba))

    logger.info(f"Precision: {np.mean(precision_scores):.4f}")
    logger.info(f"Recall: {np.mean(recall_scores):.4f}")
    logger.info(f"F1-Score: {np.mean(f1_scores):.4f}")
    logger.info(f"ROC-AUC: {np.mean(roc_auc_scores):.4f}")

    # Save the best model
    joblib.dump(best_model, 'best_model.pkl')
    logger.info("Best model saved as 'best_model.pkl'.")

    return best_model

# Function to get news sentiment
def get_news_sentiment(ticker):
    print(f"Fetching news sentiment for {ticker}")
    # Implement news sentiment fetching if required
    # Placeholder return value
    return 0

# Function to evaluate stock using the best model
def evaluate_stock_ml(ticker, best_model):
    print(f"Evaluating stock: {ticker}")
    data = yf.download(ticker, period='2mo', interval='1d')
    data.dropna(inplace=True)

    # Technical indicators
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    macd_indicator = MACD(data['Close'])
    data['MACD'] = macd_indicator.macd()
    data['MACD_Signal'] = macd_indicator.macd_signal()
    data['MACD_Hist'] = macd_indicator.macd_diff()
    data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()

    # Fill NaN values
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)

    latest_data = data.iloc[-1:]
    latest_features = latest_data[['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR']]

    # Handle missing values
    latest_features.fillna(method='ffill', inplace=True)
    latest_features.fillna(method='bfill', inplace=True)

    # Predict
    prediction = best_model.predict(latest_features)
    predicted_probability = best_model.predict_proba(latest_features)[0][1]

    print(f"Prediction for {ticker}: {prediction[0]}, Probability: {predicted_probability}")

    return prediction[0], predicted_probability, latest_data['ATR'].values[0]

# Function to calculate position size
def calculate_position_size(ticker, portfolio_value, atr):
    print(f"Calculating position size for {ticker}")
    account = api.get_account()
    buying_power = float(account.buying_power)

    # Risk per trade
    risk_per_trade = portfolio_value * MAX_RISK_PER_TRADE

    # ATR-based stop loss distance
    last_price = yf.download(ticker, period='1d', interval='1m')['Close'][-1]
    stop_loss_distance = atr * TRAILING_STOP_MULTIPLIER
    position_size = risk_per_trade / stop_loss_distance

    # Ensure position does not exceed max allocation or buying power
    max_position_value = portfolio_value * MAX_STOCK_ALLOCATION
    max_shares_based_on_allocation = max_position_value / last_price
    available_shares = buying_power / last_price
    final_position_size = int(min(position_size, max_shares_based_on_allocation, available_shares))

    print(f"Final position size for {ticker}: {final_position_size} shares")
    return final_position_size, stop_loss_distance

# Function to place limit order
def place_limit_order(ticker, position_size, action):
    print(f"Placing limit order for {ticker}")
    last_price = yf.download(ticker, period='1d', interval='1m')['Close'][-1]
    limit_price = last_price * (0.995 if action == 'buy' else 1.005)

    try:
        api.submit_order(
            symbol=ticker,
            qty=position_size,
            side=action,
            type='limit',
            time_in_force='gtc',
            limit_price=limit_price
        )
        print(f"{action.capitalize()}ing {position_size} shares of {ticker} at limit price {limit_price}")
        log_decision({"time": datetime.now().isoformat(), "ticker": ticker, "action": action.upper(), "qty": position_size, "limit_price": limit_price})
    except Exception as e:
        print(f"Error placing limit order for {ticker}: {e}")

# Function to set stop loss and take profit orders
def set_stop_loss_take_profit(ticker, position_size, stop_loss_price, take_profit_price):
    try:
        side = 'sell' if position_size > 0 else 'buy'
        api.submit_order(
            symbol=ticker,
            qty=position_size,
            side=side,
            type='stop_limit',
            time_in_force='gtc',
            stop_price=stop_loss_price,
            limit_price=stop_loss_price * 0.995
        )
        print(f"Stop loss set at {stop_loss_price}")

        api.submit_order(
            symbol=ticker,
            qty=position_size,
            side=side,
            type='limit',
            time_in_force='gtc',
            limit_price=take_profit_price
        )
        print(f"Take profit set at {take_profit_price}")

    except Exception as e:
        print(f"Error setting stop loss/take profit for {ticker}: {e}")

# Function to check portfolio and drawdown
def check_portfolio():
    account = api.get_account()
    portfolio_value = float(account.equity)
    print(f"Portfolio value: ${portfolio_value}")

    # Calculate drawdown
    historical_data = api.get_portfolio_history(period='1M')
    equity_values = historical_data.equity
    peak = max(equity_values)
    drawdown = (peak - portfolio_value) / peak
    print(f"Current drawdown: {drawdown * 100:.2f}%")

    if drawdown > MAX_DRAW_DOWN_LIMIT:
        print("Maximum drawdown exceeded. Stopping trading.")
        api.close_all_positions()
        return False
    else:
        return True

# Main trading function
def trade(best_model):
    print("Starting trading session...")
    account = api.get_account()
    portfolio_value = float(account.equity)

    for ticker in tickers:
        try:
            prediction, probability, atr = evaluate_stock_ml(ticker, best_model)
            if prediction == 1 and probability > 0.6:
                # Check if already in position
                positions = api.list_positions()
                in_position = any(p.symbol == ticker for p in positions)
                if not in_position:
                    position_size, stop_loss_distance = calculate_position_size(ticker, portfolio_value, atr)
                    if position_size > 0:
                        # Place limit buy order
                        place_limit_order(ticker, position_size, 'buy')
                        last_price = yf.download(ticker, period='1d', interval='1m')['Close'][-1]
                        stop_loss_price = last_price - stop_loss_distance
                        take_profit_price = last_price + stop_loss_distance * 2
                        set_stop_loss_take_profit(ticker, position_size, stop_loss_price, take_profit_price)
            else:
                # Check if we have a position to close
                positions = api.list_positions()
                for position in positions:
                    if position.symbol == ticker:
                        # Place limit sell order
                        position_size = abs(int(position.qty))
                        place_limit_order(ticker, position_size, 'sell')
                        print(f"Closing position for {ticker}")
        except Exception as e:
            logger.error(f"Error during trading for {ticker}: {e}")
            send_email(f"Trading Error for {ticker}", str(e))

# Schedule trading at regular intervals
def run_trading_bot():
    if check_portfolio():
        best_model = joblib.load('best_model.pkl')  # Load the pre-trained model
        trade(best_model)

# Run the bot every hour between market open and close
def schedule_trading():
    while True:
        current_time = datetime.now(eastern)
        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)

        if market_open.time() <= current_time.time() <= market_close.time():
            run_trading_bot()
        else:
            print("Market is closed. Waiting for market to open...")
        time.sleep(3600)  # Sleep for one hour

if __name__ == '__main__':
    try:
        # First, train and evaluate the model
        best_model = train_and_evaluate_model()
        # Then, start the live trading bot
        schedule_trading()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        send_email("Bot Crashed", str(e))
