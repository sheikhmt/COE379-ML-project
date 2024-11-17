import pandas as pd
import numpy as np
import yfinance as yf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import finnhub
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from scipy import stats
from secret import FINHUB_KEY
import scipy
from datetime import datetime


# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINHUB_KEY)

# Load FinBERT for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


def get_sentiment_scores(start_date, end_date, company="RTX"):
    news_data = finnhub_client.company_news(company, _from=start_date, to=end_date)
    summaries = [item["summary"] for item in news_data]

    # Convert datetime from Unix timestamp to a formatted date string
    dates = [
        datetime.fromtimestamp(item["datetime"]).strftime("%Y-%m-%d")
        for item in news_data
    ]

    sentiment_scores = []
    sentiment_dates = []
    for summary, date in zip(summaries, dates):
        with torch.no_grad():
            input_sequence = tokenizer(
                summary,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            logits = model_finbert(**input_sequence).logits
            scores = scipy.special.softmax(logits.numpy().squeeze())
            sentiment_score = scores[2] - scores[0]  # Net positive - negative score
            sentiment_scores.append(sentiment_score)
            sentiment_dates.append(date)

    # Create DataFrame with dates as index
    sentiment_df = pd.DataFrame(
        {"Sentiment": sentiment_scores}, index=pd.to_datetime(sentiment_dates)
    )

    # Aggregate daily sentiment scores by calculating the mean
    daily_scores = sentiment_df.resample("D").mean()
    daily_scores = daily_scores.dropna()

    return daily_scores


# Fetch stock data and preprocess
def fetch_stock_data(ticker, start_date, end_date):
    ticker_data = yf.Ticker(ticker)
    stock_df = ticker_data.history(start=start_date, end=end_date)
    stock_df.index = stock_df.index.tz_localize(None)
    return stock_df


# Prepare features with additional lagged sentiment values and smoothing
def prepare_features(stock_df, sentiment_df, window=3):
    sentiment_df = sentiment_df.reindex(stock_df.index).interpolate()
    stock_df["Close_Smoothed"] = stock_df["Close"].rolling(window=window).mean()
    sentiment_df["Sentiment_Smoothed"] = (
        sentiment_df["Sentiment"].rolling(window=window).mean()
    )

    data = pd.DataFrame(
        {
            "Close": stock_df["Close_Smoothed"],
            "Sentiment": sentiment_df["Sentiment_Smoothed"],
        },
        index=stock_df.index,
    ).dropna()

    data["Price_Change"] = data["Close"].pct_change()
    data["Sentiment_Change"] = data["Sentiment"].pct_change()

    # Include lagged features for sentiment
    for lag in [1, 2]:
        data[f"Sentiment_Lag_{lag}"] = data["Sentiment"].shift(lag)

    return data.dropna()


# Calculate Spearman correlation
def calculate_spearman_correlation(data):
    corr, p_value = stats.spearmanr(data["Price_Change"], data["Sentiment_Change"])
    print(
        f"Spearman correlation between sentiment and price change: {corr}, p-value: {p_value}"
    )


# Train model and make predictions
def train_and_predict(data):
    X = data[
        ["Sentiment", "Sentiment_Change", "Sentiment_Lag_1", "Sentiment_Lag_2", "Close"]
    ]
    y = data["Close"].shift(-1)

    # Train on all data until the last row
    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the next day's price
    y_pred = model.predict(X.iloc[[-1]])
    print(f"Predicted price for 10/25: {y_pred[0]}")

    return y.iloc[-1:], y_pred


# Plot results
def plot_results(data, y_test, y_pred):
    fig = go.Figure()

    # Actual prices
    fig.add_trace(
        go.Scatter(
            x=data.index[:-1],
            y=data["Close"][:-1],
            mode="lines",
            name="Actual Price",
            line=dict(color="blue"),
        )
    )

    # Predicted price for 10/25
    fig.add_trace(
        go.Scatter(
            x=[data.index[-1] + pd.Timedelta(days=1)],
            y=y_pred,
            mode="markers",
            name="Predicted Price (10/25)",
            marker=dict(color="orange", size=10),
        )
    )

    fig.update_layout(
        title="RTX Stock Price Prediction",
        xaxis_title="Date",
        yaxis_title="Stock Close Price",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )

    fig.show()


# Main function
def main():
    start_date, end_date = "2024-10-01", "2024-10-26"
    sentiment_df = get_sentiment_scores(start_date, end_date)
    stock_df = fetch_stock_data("RTX", start_date, end_date)

    data = prepare_features(stock_df, sentiment_df)
    calculate_spearman_correlation(data)
    y_test, y_pred = train_and_predict(data)

    # Check and print actual close price for 10/25 if available
    actual_close_10_25 = (
        stock_df.loc["2024-10-25"]["Close"] if "2024-10-25" in stock_df.index else None
    )
    print(f"Actual price for 10/25: {actual_close_10_25}")

    if actual_close_10_25:
        error = abs(y_pred[0] - actual_close_10_25)
        print(f"Prediction error for 10/25: {error}")

    plot_results(data, y_test, y_pred)


if __name__ == "__main__":
    main()
