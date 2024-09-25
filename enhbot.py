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
from alpaca_trade_api.stream import Stream
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
import backtrader as bt
import xgboost as xgb
import joblib

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

# Initialize Alpaca API and WebSocket Stream
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
stream = Stream(API_KEY, API_SECRET, WS_URL, data_feed='iex')

# Sentiment Analyzer (Consider replacing with a financial-specific model)
analyzer = SentimentIntensityAnalyzer()

# Risk parameters
INITIAL_PORTFOLIO_VALUE = 30000  # Updated initial portfolio value
MAX_RISK_PER_TRADE = 0.005  # Risk 0.5% of the portfolio per trade
MAX_DRAW_DOWN_LIMIT = 0.05  # Stop trading if drawdown exceeds 5%
MAX_STOCK_ALLOCATION = 0.10  # Maximum 10% of portfolio in one stock
TRAILING_STOP_MULTIPLIER = 3  # ATR multiplier for stop loss

# Transaction cost assumptions
TRANSACTION_COST = 0.0005  # 0.05% per trade

# JSON log file
LOG_FILE = 'trading_log.json'

# Define Eastern Time Zone
eastern = pytz.timezone('US/Eastern')

# List of tickers (Diversified universe)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'XOM', 'JNJ', 'WMT']

# Backtesting parameters
START_DATE = '2022-01-01'
END_DATE = '2023-01-01'
INITIAL_CAPITAL = 27000
COMMISSION = 0.0000  # 0.00%

# Logging setup
logger = logging.getLogger('TradingBot')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('trading_bot.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Machine Learning Model (will be set after hyperparameter tuning)
best_model = None

# Machine Learning Model (Placeholder)
ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
scaler = StandardScaler()

# Global variables
position_sizes = {}

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
            print("Error notification sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to log decisions
def log_decision(log_entry):
    log_file = "trade_decisions.json"

    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([], f)

    with open(log_file, 'r') as f:
        logs = json.load(f)
        if not isinstance(logs, list):
            logs = []

    logs.append(log_entry)

    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)

    print(f"Logged decision for {log_entry['ticker']} action: {log_entry['action']}")

# Function to calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.95):
    var = np.percentile(returns, (1 - confidence_level) * 100)
    return var * np.sqrt(10)  # 10-day VaR

# Function to get news sentiment
def get_news_sentiment(ticker):
    print(f"Fetching news sentiment for {ticker}")
    url = f'https://finnhub.io/api/v1/news?category=company&symbol={ticker}&token={NEWS_API_KEY}'

    response = requests.get(url)
    news_data = response.json()

    if response.status_code == 200 and news_data:
        total_score = 0
        for article in news_data:
            if 'headline' in article:
                sentiment_score = analyzer.polarity_scores(article['headline'])
                total_score += sentiment_score['compound']
        avg_sentiment = total_score / len(news_data)
        print(f"Average sentiment for {ticker}: {avg_sentiment}")
        return avg_sentiment
    else:
        print(f"No news data available for {ticker}")
        return 0

# Feature Engineering Function
def prepare_ml_data(ticker):
    data = yf.download(ticker, start=START_DATE, end=END_DATE)
    data.dropna(inplace=True)

    # Technical Indicators
    data['RSI'] = bt.indicators.RSI(bt.feeds.PandasData(dataname=data), period=14).array
    data['MACD'] = bt.indicators.MACD(bt.feeds.PandasData(dataname=data)).macd.array
    data['MACD_Signal'] = bt.indicators.MACD(bt.feeds.PandasData(dataname=data)).signal.array
    data['ATR'] = bt.indicators.ATR(bt.feeds.PandasData(dataname=data)).array

    # Fill NaN values
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)

    # Target Variable
    data['Future_Return'] = data['Close'].shift(-1) / data['Close'] - 1
    data['Target'] = (data['Future_Return'] > COMMISSION).astype(int)

    # Features and Labels
    features = data[['RSI', 'MACD', 'MACD_Signal', 'ATR']]
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

# Function to train ML model with hyperparameter tuning
def train_ml_model():
    print("Training machine learning model with hyperparameter tuning...")
    all_features = pd.DataFrame()
    all_labels = pd.Series(dtype=int)

    for ticker in tickers:
        features, labels = prepare_ml_data(ticker)
        all_features = all_features.append(features)
        all_labels = all_labels.append(labels)

    # Handle missing values
    all_features.fillna(method='ffill', inplace=True)
    all_features.fillna(method='bfill', inplace=True)

    # Hyperparameter tuning
    best_model = hyperparameter_tuning(all_features, all_labels)

    return best_model

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
    bb_indicator = BollingerBands(data['Close'])
    data['BB_High'] = bb_indicator.bollinger_hband()
    data['BB_Low'] = bb_indicator.bollinger_lband()
    data['BB_Middle'] = bb_indicator.bollinger_mavg()
    data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()

    # Fundamental data
    stock_info = yf.Ticker(ticker).info
    data['PE_Ratio'] = stock_info.get('trailingPE', np.nan)
    data['PB_Ratio'] = stock_info.get('priceToBook', np.nan)

    # Sentiment score
    data['Sentiment'] = get_news_sentiment(ticker)

    # Additional features
    data['Price_Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()

    data.dropna(inplace=True)

    latest_data = data.iloc[-1:]
    latest_features = latest_data[['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_High', 'BB_Low', 'BB_Middle', 'ATR', 'PE_Ratio', 'PB_Ratio', 'Sentiment', 'Price_Change', 'Volume_Change']]

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
        api.submit_order(
            symbol=ticker,
            qty=position_size,
            side='sell' if position_size > 0 else 'buy',
            type='stop_limit',
            time_in_force='gtc',
            stop_price=stop_loss_price,
            limit_price=stop_loss_price * 0.995
        )
        print(f"Stop loss set at {stop_loss_price}")

        api.submit_order(
            symbol=ticker,
            qty=position_size,
            side='sell' if position_size > 0 else 'buy',
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

# Schedule trading at regular intervals
def run_trading_bot():
    if check_portfolio():
        best_model = train_ml_model()
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

# Backtrader Strategy Class
class MLStrategy(bt.Strategy):
    params = (
        ('model', None),
        ('ticker', None),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.model = self.params.model
        self.ticker = self.params.ticker

        # Indicators
        self.rsi = bt.indicators.RSI(self.datas[0], period=14)
        self.macd = bt.indicators.MACD(self.datas[0])
        self.atr = bt.indicators.ATR(self.datas[0])

    def next(self):
        # Feature Vector
        features = pd.DataFrame({
            'RSI': [self.rsi[0]],
            'MACD': [self.macd.macd[0]],
            'MACD_Signal': [self.macd.signal[0]],
            'ATR': [self.atr[0]],
        })

        # Handle missing values
        features.fillna(method='ffill', inplace=True)
        features.fillna(method='bfill', inplace=True)

        # Prediction
        prediction = self.model.predict(features)
        predicted_probability = self.model.predict_proba(features)[0][1]

        # Risk Management Parameters
        position_size = self.broker.getvalue() * 0.01 / self.atr[0]  # 1% of capital per ATR
        position_size = int(position_size / self.dataclose[0])  # Number of shares

        # Check if we are in the market
        if not self.position:
            if prediction == 1 and predicted_probability > 0.6:
                self.buy(size=position_size)
                logger.info(f"Buying {position_size} shares of {self.ticker} at {self.dataclose[0]}")
        else:
            if prediction == 0:
                self.sell(size=self.position.size)
                logger.info(f"Selling {self.position.size} shares of {self.ticker} at {self.dataclose[0]}")

# Backtesting Function
def backtest_strategy():
    logger.info("Starting backtest...")
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION)

    # Load the best model
    best_model = joblib.load('best_model.pkl')

    for ticker in tickers:
        data = bt.feeds.YahooFinanceData(
            dataname=ticker,
            fromdate=pd.to_datetime(START_DATE),
            todate=pd.to_datetime(END_DATE)
        )
        cerebro.adddata(data, name=ticker)
        cerebro.addstrategy(MLStrategy, model=best_model, ticker=ticker)

    # Run backtest
    results = cerebro.run()
    final_portfolio_value = cerebro.broker.getvalue()
    logger.info(f"Final Portfolio Value: ${final_portfolio_value:.2f}")

    # Calculate Performance Metrics
    pnl = final_portfolio_value - INITIAL_CAPITAL
    logger.info(f"Total PnL: ${pnl:.2f}")

    # Sharpe Ratio
    analyzer = bt.analyzers.SharpeRatio_A
    cerebro.addanalyzer(analyzer, _name='sharpe')
    strat = results[0]
    sharpe_ratio = strat.analyzers.sharpe.get_analysis()['sharperatio']
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    # Maximum Drawdown
    analyzer = bt.analyzers.DrawDown
    cerebro.addanalyzer(analyzer, _name='drawdown')
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown.max.drawdown
    logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")

    # Plotting
    cerebro.plot()

if __name__ == '__main__':
    try:
        schedule_trading()
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        send_email("Bot Crashed", str(e))
