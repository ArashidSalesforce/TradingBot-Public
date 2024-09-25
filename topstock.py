import yfinance as yf
import pandas as pd

# SP100 stock list
sp100_stocks = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'AIG', 'ALL', 'AMGN', 'AMT', 'AMZN', 'AXP', 
    'BA', 'BAC', 'BIIB', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK.B', 'C', 'CAT', 
    'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 
    'DD', 'DE', 'DHR', 'DIS', 'DOW', 'DUK', 'EMR', 'EXC', 'F', 'FDX', 'GD', 
    'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC', 
    'JNJ', 'JPM', 'KHC', 'KMI', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 
    'MCD', 'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 
    'NEE', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PM', 'PYPL', 
    'QCOM', 'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TMUS', 
    'TSLA', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WBA', 'WFC', 
    'WMT', 'XOM'
]

# Function to get today's stock performance
def get_stock_performance_today(stock_list):
    stock_performance = {}
    for stock in stock_list:
        print(f"Fetching data for {stock}...")
        # Download today's stock data (interval can be set to '1d' or intraday like '1m')
        data = yf.download(stock, period='1d', interval='1m')  # Adjust interval as needed
        if not data.empty:
            # Calculate the percentage change from today's open to the most recent price
            performance = (data['Close'].iloc[-1] - data['Open'].iloc[0]) / data['Open'].iloc[0]
            stock_performance[stock] = performance
            print(f"{stock}: Performance calculated as {performance:.2%}")
        else:
            print(f"No data available for {stock}. Skipping.")
    return stock_performance

# Function to get top N performing stocks for today
def get_top_n_stocks_today(stock_list, n=3):
    print("Calculating today's stock performances...")
    performance = get_stock_performance_today(stock_list)
    top_stocks = sorted(performance, key=performance.get, reverse=True)[:n]
    print(f"Top {n} performing stocks today: {top_stocks}")
    return top_stocks

# Get the top-performing stock today from SP100 list
top_stock_today = get_top_n_stocks_today(sp100_stocks, n=3)
print("Final top-performing stock today:", top_stock_today)


