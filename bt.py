import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

class MyStrategy(bt.Strategy):
    params = (
        ('max_risk_per_trade', 0.01),  # 1% of portfolio per trade
        ('max_stock_allocation', 0.20),  # 20% of portfolio in one stock
        ('trailing_stop_percent', 0.0001),  # 0.1% trailing stop
    )

    def __init__(self):
        self.position_highest_price = {}
        self.order = None  # Keep track of pending orders
        self.data_indicators = {}

        # Pre-compute indicators for each data feed
        for data in self.datas:
            self.data_indicators[data] = {}
            self.data_indicators[data]['rsi'] = bt.indicators.RSI(data.close, period=14)
            self.data_indicators[data]['ema_20'] = bt.indicators.ExponentialMovingAverage(data.close, period=20)
            self.data_indicators[data]['ema_50'] = bt.indicators.ExponentialMovingAverage(data.close, period=50)

    def next(self):
        for data in self.datas:
            ticker = data._name
            pos = self.getposition(data)
            rsi = self.data_indicators[data]['rsi'][0]
            ema_20 = self.data_indicators[data]['ema_20'][0]
            ema_50 = self.data_indicators[data]['ema_50'][0]
            close_price = data.close[0]

            # Decision logic without sentiment score
            decision = "NO_ACTION"
            if rsi > 50 and ema_20 > ema_50:
                decision = "BUY"
            elif rsi < 50 and ema_20 < ema_50:
                decision = "SELL"

            # Execute trades
            if not pos:
                if decision == "BUY":
                    # Calculate position size
                    cash = self.broker.getcash()
                    portfolio_value = self.broker.getvalue()
                    risk_per_trade = portfolio_value * self.params.max_risk_per_trade
                    var = self.calculate_var(data)
                    if var == 0:
                        continue
                    position_size = risk_per_trade / (close_price * abs(var))
                    max_shares = (portfolio_value * self.params.max_stock_allocation) / close_price
                    final_position_size = int(min(position_size, max_shares))

                    if final_position_size > 0:
                        self.order = self.buy(data=data, size=final_position_size)
                        self.position_highest_price[ticker] = close_price
                        print(f"Buying {final_position_size} shares of {ticker} at {close_price}")
            else:
                # Update highest price
                if ticker not in self.position_highest_price:
                    self.position_highest_price[ticker] = close_price
                if close_price > self.position_highest_price[ticker]:
                    self.position_highest_price[ticker] = close_price

                # Check trailing stop loss
                trailing_stop_price = self.position_highest_price[ticker] * (1 - self.params.trailing_stop_percent)
                if close_price < trailing_stop_price:
                    self.order = self.close(data=data)
                    del self.position_highest_price[ticker]
                    print(f"Selling {pos.size} shares of {ticker} at {close_price} due to trailing stop loss")

    def calculate_var(self, data):
        # Calculate historical VaR
        close_prices = np.array(data.close.get(size=15))
        if len(close_prices) < 15:
            return 0
        returns = np.log(close_prices[1:] / close_prices[:-1])
        var = np.percentile(returns, 5)  # 95% confidence level
        return abs(var)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED: {order.executed.size} shares at {order.executed.price}")
            elif order.issell():
                print(f"SELL EXECUTED: {order.executed.size} shares at {order.executed.price}")
            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('Order Canceled/Margin/Rejected')
            self.order = None

# Prepare data feeds
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
data_feeds = []

for ticker in tickers:
    df = yf.download(ticker, start='2023-01-01', end='2024-08-31')
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        data = bt.feeds.PandasData(dataname=df, name=ticker)
        data_feeds.append(data)

# Set up Cerebro engine
cerebro = bt.Cerebro()
cerebro.addstrategy(MyStrategy)

for data in data_feeds:
    cerebro.adddata(data)

cerebro.broker.setcash(26000.0)

print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())
cerebro.run()
print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
