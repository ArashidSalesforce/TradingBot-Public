import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import backtrader as bt
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import logging
import joblib
import os

# Import the 'ta' library indicators
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

logging.basicConfig(level=logging.INFO)

# Global variables
tickers = ['TSLA']

# Backtesting parameters
START_DATE = '2015-01-01'
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
    data['SMA'] = data['Close'].rolling(window=14).mean()
    data['EMA'] = data['Close'].ewm(span=14, adjust=False).mean()
    bollinger = BollingerBands(close=data['Close'])
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Low'] = bollinger.bollinger_lband()
    stochastic = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
    data['Stochastic'] = stochastic.stoch()
    data['OBV'] = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()

    # Fill NaN values
    data.ffill(inplace=True)
    data.dropna(inplace=True)

    # Target Variable
    data['Future_Return'] = data['Close'].shift(-1) / data['Close'] - 1

    # Original labels: -1 (Sell Short), 0 (Hold), 1 (Buy)
    # Mapped labels: 0 (Sell Short), 1 (Hold), 2 (Buy)
    def map_target(x):
        if x > COMMISSION:
            return 2  # Buy
        elif x < -COMMISSION:
            return 0  # Sell Short
        else:
            return 1  # Hold

    data['Target'] = data['Future_Return'].apply(map_target)

    # Features and Labels
    features = data[['RSI', 'MACD', 'MACD_Signal', 'ATR', 'SMA', 'EMA', 'BB_High', 'BB_Low', 'Stochastic', 'OBV']]
    labels = data['Target']

    return features[:-1], labels[:-1]  # Exclude the last row due to NaN in 'Future_Return'

# Hyperparameter Tuning Function
def hyperparameter_tuning(X, y):
    logger.info("Starting hyperparameter tuning...")

    classifiers = {
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss')  # Removed use_label_encoder
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
            scoring='f1_macro',
            cv=tscv,
            n_jobs=-1,
            random_state=42,
            error_score='raise'  # Raise errors to handle them
        )

        try:
            search.fit(X, y)
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_model = search.best_estimator_
                best_params = search.best_params_
        except ValueError as e:
            logger.warning(f"Model {clf_name} failed with error: {e}")

    if best_model is None:
        logger.error("No suitable model found during hyperparameter tuning.")
        raise Exception("Hyperparameter tuning failed.")

    logger.info(f"Best Model: {best_model.named_steps['classifier'].__class__.__name__}")
    logger.info(f"Best Score: {best_score}")
    logger.info(f"Best Parameters: {best_params}")

    return best_model

# Function to Train ML Model and Evaluate
def train_and_evaluate_model():
    logger.info("Training and evaluating the machine learning model...")
    features, labels = prepare_ml_data('TSLA')

    # Handle missing values
    features.ffill(inplace=True)
    features.bfill(inplace=True)

    # Hyperparameter Tuning
    best_model = hyperparameter_tuning(features, labels)

    # Model Evaluation Metrics
    tscv = TimeSeriesSplit(n_splits=5)
    f1_scores = []

    for train_index, test_index in tscv.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        f1_scores.append(f1_score(y_test, y_pred, average='macro'))

    logger.info(f"F1-Score: {np.mean(f1_scores):.4f}")

    # Save the best model
    joblib.dump(best_model, 'best_model.pkl')
    logger.info("Best model saved as 'best_model.pkl'.")

    return best_model

# Custom OnBalanceVolume Indicator
class OnBalanceVolume(bt.Indicator):
    lines = ('obv',)
    plotinfo = dict(subplot=True)

    def __init__(self):
        self.addminperiod(2)
        # Removed self.lines.obv assignment

    def next(self):
        if len(self) == 1:
            self.lines.obv[0] = self.data.volume[0]
        else:
            if self.data.close[0] > self.data.close[-1]:
                delta = self.data.volume[0]
            elif self.data.close[0] < self.data.close[-1]:
                delta = -self.data.volume[0]
            else:
                delta = 0
            self.lines.obv[0] = self.lines.obv[-1] + delta

# Backtesting Function
def backtest_strategy():
    logger.info("Starting backtest...")
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=COMMISSION)

    # Load the best model
    best_model = joblib.load('best_model.pkl')

    # Fetch data using yfinance
    data = yf.download('TSLA', start=START_DATE, end=END_DATE)
    if data.empty:
        logger.warning("No data found for TSLA. Exiting.")
        return
    data.index.name = 'datetime'
    data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    data['openinterest'] = 0
    data_bt = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_bt)

    cerebro.addstrategy(MLStrategy, model=best_model)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

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
    max_drawdown = drawdown.max.drawdown
    logger.info(f"Sharpe Ratio: {sharpe_ratio}")
    logger.info(f"Maximum Drawdown: {max_drawdown:.2f}%")

    # Save the plot to a file
    fig = cerebro.plot()[0][0]
    fig.savefig('backtest_results.png')
    logger.info("Backtest plot saved as 'backtest_results.png'.")

class MLStrategy(bt.Strategy):
    params = (
        ('model', None),
    )

    def __init__(self):
        self.model = self.params.model
        self.dataclose = self.datas[0].close

        # Indicators
        self.rsi = bt.indicators.RSI(self.datas[0], period=14)
        self.macd = bt.indicators.MACD(self.datas[0])
        self.atr = bt.indicators.ATR(self.datas[0])
        self.sma = bt.indicators.SMA(self.datas[0], period=14)
        self.ema = bt.indicators.EMA(self.datas[0], period=14)
        self.bollinger = bt.indicators.BollingerBands(self.datas[0])
        self.stochastic = bt.indicators.StochasticSlow(self.datas[0])
        self.obv = OnBalanceVolume(self.datas[0])  # Use custom indicator

    def next(self):
        # Feature Vector
        features = pd.DataFrame({
            'RSI': [self.rsi[0]],
            'MACD': [self.macd.macd[0]],
            'MACD_Signal': [self.macd.signal[0]],
            'ATR': [self.atr[0]],
            'SMA': [self.sma[0]],
            'EMA': [self.ema[0]],
            'BB_High': [self.bollinger.top[0]],
            'BB_Low': [self.bollinger.bot[0]],
            'Stochastic': [self.stochastic.percK[0]],
            'OBV': [self.obv.obv[0]],  # Corrected line
        })

        # Handle missing values
        features.ffill(inplace=True)
        features.bfill(inplace=True)

        # Prediction
        prediction = self.model.predict(features)[0]
        predicted_proba = self.model.predict_proba(features)[0]

        position = self.getposition()

        # Risk Management Parameters
        if self.atr[0] != 0:
            position_size = self.broker.getvalue() * 0.01 / self.atr[0]
            position_size = int(position_size / self.dataclose[0])
        else:
            position_size = 0

        # Trading Logic
        if not position:
            if prediction == 2 and position_size > 0:
                self.buy(size=position_size)
                logger.info(f"Buying {position_size} shares at {self.dataclose[0]}")
            elif prediction == 0 and position_size > 0:
                self.sell(size=position_size)
                logger.info(f"Shorting {position_size} shares at {self.dataclose[0]}")
        else:
            if position.size > 0:  # Currently in a long position
                if prediction == 0:
                    self.close()
                    self.sell(size=position_size)
                    logger.info(f"Closing long position and entering short at {self.dataclose[0]}")
                elif prediction == 1:
                    self.close()
                    logger.info(f"Closing long position at {self.dataclose[0]}")
                # Else, prediction == 2 (continue holding or add to position)
            elif position.size < 0:  # Currently in a short position
                if prediction == 2:
                    self.close()
                    self.buy(size=position_size)
                    logger.info(f"Closing short position and entering long at {self.dataclose[0]}")
                elif prediction == 1:
                    self.close()
                    logger.info(f"Closing short position at {self.dataclose[0]}")
                # Else, prediction == 0 (continue holding or add to position)

if __name__ == '__main__':
    try:
        best_model = train_and_evaluate_model()
        backtest_strategy()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
