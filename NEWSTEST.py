import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

NEWS_API_KEY = ''

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


if __name__ == '__main__':
    get_news_sentiment('TSLA')