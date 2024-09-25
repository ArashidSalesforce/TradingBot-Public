import backtrader as bt
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import logging
import joblib
import os
import json

# Import the 'ta' library indicators
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

logging.basicConfig(level=logging.INFO)

# Global variables
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'XOM', 'JNJ', 'WMT']

# Backtesting parameters
START_DATE = '2019-01-01'  # For example, start from 2019
END_DATE = '2023-01-01'
INITIAL_CAPITAL = 30000
COMMISSION = 0.0000  # 0.00%

# Logging setup
logger = logging.getLogger('TradingBot')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('trading_bot.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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
    data = yf.download(ticker, start=START_DATE, end=END_DATE)
    data.dropna(inplace=True)

    # Technical Indicators using 'ta' library
    data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()

    # Fill NaN values
    data.ffill(inplace=True)
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
        'XGBoost': xgb.XGBClassifier(eval_metric='logloss')
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
    features_list = []
    labels_list = []

    for ticker in tickers:
        features, labels = prepare_ml_data(ticker)
        features_list.append(features)
        labels_list.append(labels)

    # Concatenate all features and labels
    all_features = pd.concat(features_list, ignore_index=True)
    all_labels = pd.concat(labels_list, ignore_index=True)

    # Handle missing values
    all_features.ffill(inplace=True)
    all_features.bfill(inplace=True)

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

# Backtesting Function
def backtest_strategy():
    logger.info("Starting backtest...")
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION)

    # Load the best model
    best_model = joblib.load('best_model.pkl')

    for ticker in tickers:
        # Fetch data using yfinance
        data = yf.download(ticker, start=START_DATE, end=END_DATE)
        if data.empty:
            logger.warning(f"No data found for {ticker}. Skipping.")
            continue
        data.index.name = 'datetime'  # Ensure index is named 'datetime' for Backtrader
        data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        data['openinterest'] = 0  # Add openinterest column required by Backtrader
        data_bt = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_bt, name=ticker)

    cerebro.addstrategy(MLStrategy, model=best_model)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')


    # Run backtest
    results = cerebro.run()
    final_portfolio_value = cerebro.broker.getvalue()
    logger.info(f"Final Portfolio Value: ${final_portfolio_value:.2f}")

    # Calculate Performance Metrics
    pnl = final_portfolio_value - INITIAL_CAPITAL
    logger.info(f"Total PnL: ${pnl:.2f}")

    # Get analyzers
    strat = results[0]
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()
    max_drawdown = drawdown.max.drawdown
    logger.info(f"Sharpe Ratio: {sharpe_ratio}")
    logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")

    # Plotting
    cerebro.plot()

class MLStrategy(bt.Strategy):
    params = (
        ('model', None),
    )

    def __init__(self):
        self.model = self.params.model
        self.dataclose = dict()
        self.rsi = dict()
        self.macd = dict()
        self.atr = dict()

        for data in self.datas:
            self.dataclose[data._name] = data.close

            # Indicators
            self.rsi[data._name] = bt.indicators.RSI(data, period=14)
            self.macd[data._name] = bt.indicators.MACD(data)
            self.atr[data._name] = bt.indicators.ATR(data)

    def next(self):
        for data in self.datas:
            ticker = data._name
            position = self.getposition(data)
            # Feature Vector
            features = pd.DataFrame({
                'RSI': [self.rsi[ticker][0]],
                'MACD': [self.macd[ticker].macd[0]],
                'MACD_Signal': [self.macd[ticker].signal[0]],
                'ATR': [self.atr[ticker][0]],
            })

            # Handle missing values
            features.ffill(inplace=True)
            features.bfill(inplace=True)

            # Prediction
            prediction = self.model.predict(features)
            predicted_probability = self.model.predict_proba(features)[0][1]

            # Risk Management Parameters
            if self.atr[ticker][0] != 0:
                position_size = self.broker.getvalue() * 0.01 / self.atr[ticker][0]  # 1% of capital per ATR
                position_size = int(position_size / self.dataclose[ticker][0])  # Number of shares
            else:
                position_size = 0

            # Check if we are in the market
            if not position:
                if prediction == 1 and predicted_probability > 0.6 and position_size > 0:
                    self.buy(data=data, size=position_size)
                    logger.info(f"Buying {position_size} shares of {ticker} at {self.dataclose[ticker][0]}")
            else:
                if prediction == 0:
                    self.sell(data=data, size=position.size)
                    logger.info(f"Selling {position.size} shares of {ticker} at {self.dataclose[ticker][0]}")

# Main execution block remains the same
if __name__ == '__main__':
    try:
        best_model = train_and_evaluate_model()
        backtest_strategy()
    except ImportError as e:
        logger.error(f"ImportError: {e}")
        print("Please ensure all required libraries are installed.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")