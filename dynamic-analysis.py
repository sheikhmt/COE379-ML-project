import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google.cloud import storage
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import scipy
from scipy import stats
from datetime import datetime, timedelta

# Initialize FinBERT for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Define bucket and directories for each company
BUCKET_NAME = "fin_analysis_data"
COMPANIES = {
    "RTX": "raytheon/news_articles",
    "HON": "honeywell/news_articles",
    "LMT": "lockheed/news_articles",
}


def pull_news_from_gcs(company, date):
    """Pull news articles for a specific company and date from GCS."""
    bucket = storage_client.bucket(BUCKET_NAME)
    file_name = f"{company}_{date}_news.json"
    blob_path = f"fin_text/{COMPANIES[company]}/{file_name}"
    print(blob_path)
    blob = bucket.blob(blob_path)

    if blob.exists():
        news_data = json.loads(blob.download_as_text())
        return news_data
    else:
        print("didn't pull data")


def analyze_sentiment(news_data):
    """Analyze sentiment of news articles using FinBERT."""
    sentiment_scores = []
    for article in news_data:
        summary = article.get("summary", "")
        if summary:
            with torch.no_grad():
                inputs = tokenizer(
                    summary,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                logits = model_finbert(**inputs).logits
                scores = scipy.special.softmax(logits.numpy().squeeze())
                sentiment_score = scores[2] - scores[0]  # Positive - Negative
                sentiment_scores.append(sentiment_score)
        else:
            sentiment_scores.append(None)
    return sentiment_scores


def get_sentiment_scores(company, start_date, end_date):
    """Aggregate sentiment scores for a date range."""
    all_sentiments = []
    all_dates = pd.date_range(start=start_date, end=end_date)

    for date in all_dates:
        news_data = pull_news_from_gcs(company, date.strftime("%Y-%m-%d"))
        sentiments = analyze_sentiment(news_data)
        if sentiments:
            avg_sentiment = sum(filter(None, sentiments)) / len(sentiments)
            all_sentiments.append((date, avg_sentiment))

    sentiment_df = pd.DataFrame(all_sentiments, columns=["Date", "Sentiment"])
    sentiment_df.set_index("Date", inplace=True)
    return sentiment_df


def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    stock_data.index = stock_data.index.tz_localize(None)
    return stock_data


def prepare_features(stock_df, sentiment_df, window=3):
    if stock_df.empty or sentiment_df.empty:
        raise ValueError("Stock or Sentiment data is empty. Check inputs.")

    # Ensure 'Sentiment' column is numeric
    sentiment_df["Sentiment"] = pd.to_numeric(
        sentiment_df["Sentiment"], errors="coerce"
    )

    # Restrict to common indices
    common_index = stock_df.index.intersection(sentiment_df.index)
    stock_df = stock_df.loc[common_index]
    sentiment_df = sentiment_df.loc[common_index]

    # Check if any data remains after alignment
    if stock_df.empty or sentiment_df.empty:
        raise ValueError("No overlapping dates between stock and sentiment data.")

    # Align sentiment data with stock data
    sentiment_df = sentiment_df.reindex(stock_df.index).interpolate()

    # Apply rolling window smoothing
    stock_df["Close_Smoothed"] = stock_df["Close"].rolling(window=window).mean()
    sentiment_df["Sentiment_Smoothed"] = (
        sentiment_df["Sentiment"].rolling(window=window).mean()
    )

    # Combine data into a single DataFrame
    data = pd.DataFrame(
        {
            "Close": stock_df["Close_Smoothed"],
            "Sentiment": sentiment_df["Sentiment_Smoothed"],
        },
        index=stock_df.index,
    ).dropna()

    # Compute percentage change for features
    data["Price_Change"] = data["Close"].pct_change()
    data["Sentiment_Change"] = data["Sentiment"].pct_change()

    # Add lagged sentiment features
    for lag in [1, 2]:
        data[f"Sentiment_Lag_{lag}"] = data["Sentiment"].shift(lag)

    # Drop rows with NaNs
    data = data.dropna()

    if data.empty:
        raise ValueError("Insufficient data after preparing features.")

    return data


def train_and_predict(data):
    """Train the model and predict the next day's price."""
    X = data[
        ["Sentiment", "Sentiment_Change", "Sentiment_Lag_1", "Sentiment_Lag_2", "Close"]
    ]
    y = data["Close"].shift(-1)

    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X.iloc[[-1]])
    return y.iloc[-1:], y_pred


def plot_results(data, y_pred):
    """Visualize predictions vs. actual stock prices."""
    fig = go.Figure()

    # Actual stock prices
    fig.add_trace(
        go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Actual Price")
    )

    # Predicted price
    future_date = data.index[-1] + timedelta(days=1)
    fig.add_trace(
        go.Scatter(x=[future_date], y=y_pred, mode="markers", name="Predicted Price")
    )

    fig.update_layout(
        title="Stock Price Prediction with Sentiment Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
    )
    fig.show()


def main():
    #needs more data uploaded to google cloud before it can start working
    start_date, end_date = "2024-10-16", "2024-11-16"
    company = "RTX"

    sentiment_data = get_sentiment_scores(company, start_date, end_date)
    stock_data = fetch_stock_data(company, start_date, end_date)

    data = prepare_features(stock_data, sentiment_data)
    y_test, y_pred = train_and_predict(data)

    print(f"Predicted price: {y_pred[0]}")
    plot_results(data, y_pred)


if __name__ == "__main__":
    main()
