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

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINHUB_KEY)

# Load FinBERT for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


# Fetch and preprocess sentiment scores
def get_sentiment_scores(start_date, end_date, company="RTX"):
    news_data = finnhub_client.company_news(company, _from=start_date, to=end_date)
    summaries = [item["summary"] for item in news_data]

    preds_proba = []
    for summary in summaries:
        with torch.no_grad():
            input_sequence = tokenizer(
                summary,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            logits = model_finbert(**input_sequence).logits
            scores = {
                k: v
                for k, v in zip(
                    model_finbert.config.id2label.values(),
                    scipy.special.softmax(logits.numpy().squeeze()),
                )
            }
            preds_proba.append(max(scores.values()))

    # Remove outliers using Z-score
    sentiment_scores = np.array(preds_proba)
    z_scores = np.abs(stats.zscore(sentiment_scores))
    sentiment_scores = sentiment_scores[z_scores < 2]

    sentiment_df = pd.DataFrame(
        {
            "Date": pd.date_range(start=start_date, periods=len(sentiment_scores)),
            "Sentiment": sentiment_scores,
        }
    )
    return sentiment_df.set_index("Date")


# Fetch stock data and preprocess
def fetch_stock_data(ticker, start_date, end_date):
    ticker_data = yf.Ticker(ticker)
    stock_df = ticker_data.history(start=start_date, end=end_date)
    print(stock_df)
    # stock_df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    stock_df.index = stock_df.index.tz_localize(None)
    return stock_df


# Combine and prepare features
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

    return data.dropna()


# Train model and make predictions, including 10/25 forecast
def train_and_predict(data):
    X = data[["Sentiment", "Sentiment_Change", "Close"]]
    y = data["Close"].shift(-1)  # Next day’s price as target

    # Use all data until 10/24 for training
    X_train, y_train = X.iloc[:-1], y.iloc[:-1]
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict 10/25 price
    y_pred = model.predict(X.iloc[[-1]])
    print(f"Predicted price for 10/25: {y_pred[0]}")

    return y.iloc[-1:], y_pred  # Return last actual and predicted values


# Plotting the results with Plotly
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
