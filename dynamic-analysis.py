import json
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google.cloud import storage
import yfinance as yf
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
import scipy
from scipy import stats
from datetime import datetime, timedelta
from google.oauth2 import service_account

# Existing credentials and initialization code remains the same
# credentials_path = "D:/Coding/IntroML/coe379-ml-project-81b7de97df4a.json"
credentials_path = "C:/Users/arish/Coding/MLClass/coe379-ml-project-55cd4cde117a.json"
credentials = service_account.Credentials.from_service_account_file(credentials_path)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
storage_client = storage.Client(credentials=credentials)

BUCKET_NAME = "fin_analysis_data"
COMPANIES = {
    "RTX": "raytheon/news_articles/to_delete",
    "NOC": "northrop/news_articles/to_delete",
    "LMT": "lockheed/news_articles/to_delete",
}


# Previous helper functions remain the same
def pull_news_from_gcs(company, date):
    """Pull news articles for a specific company and date from GCS."""
    bucket = storage_client.bucket(BUCKET_NAME)
    file_name = f"{company}_{date}_news.json"
    blob_path = f"fin_text/{COMPANIES[company]}/{file_name}"
    blob = bucket.blob(blob_path)

    if blob.exists():
        news_data = json.loads(blob.download_as_text())
        if not news_data:
            print(f"No news data available in the file for {company} on {date}")
        return news_data
    else:
        print(f"No blob found for {company} on {date} at path {blob_path}")
        return []


def analyze_sentiment(news_data):
    """Analyze sentiment of news articles using FinBERT with confidence scores."""
    sentiment_scores = []
    confidence_scores = []

    for article in news_data:
        headline = article.get("headline", "")
        summary = article.get("summary", "")
        if summary and headline:
            with torch.no_grad():
                text_in = summary + "\n" + headline
                inputs = tokenizer(
                    text_in,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                logits = model_finbert(**inputs).logits
                scores = scipy.special.softmax(logits.numpy().squeeze())
                sentiment_score = scores[2] - scores[0]  # Positive - Negative
                confidence_score = np.max(
                    scores
                )  # Confidence is the highest probability
                sentiment_scores.append(sentiment_score)
                confidence_scores.append(confidence_score)

    return sentiment_scores, confidence_scores


def get_sentiment_scores(company, start_date, end_date):
    """Aggregate sentiment scores with confidence weights for a date range."""
    all_sentiments = []
    all_dates = pd.date_range(start=start_date, end=end_date)

    for date in all_dates:
        news_data = pull_news_from_gcs(company, date.strftime("%Y-%m-%d"))
        if news_data:
            sentiments, confidences = analyze_sentiment(news_data)
            if sentiments:
                # Weight sentiments by their confidence scores
                weighted_sentiment = np.average(sentiments, weights=confidences)
                all_sentiments.append((date, weighted_sentiment))

    sentiment_df = pd.DataFrame(all_sentiments, columns=["Date", "Sentiment"])
    sentiment_df.set_index("Date", inplace=True)
    return sentiment_df


def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data with additional technical indicators."""
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Add more buffer days for calculating technical indicators
    buffer_start = (start_dt - timedelta(days=30)).strftime("%Y-%m-%d")
    buffer_end = end_dt.strftime("%Y-%m-%d")

    stock_data = yf.Ticker(ticker).history(start=buffer_start, end=buffer_end)
    stock_data.index = stock_data.index.tz_localize(None)

    # Calculate technical indicators
    stock_data["SMA_5"] = stock_data["Close"].rolling(window=5).mean()
    stock_data["SMA_20"] = stock_data["Close"].rolling(window=20).mean()
    stock_data["RSI"] = calculate_rsi(stock_data["Close"])
    stock_data["Volatility"] = stock_data["Close"].rolling(window=5).std()

    return stock_data


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def prepare_features(stock_df, sentiment_df, window=2):  # Reduced window size
    """Prepare features with enhanced technical indicators and sentiment analysis."""
    # Basic data preparation
    sentiment_df = sentiment_df[~sentiment_df.index.duplicated(keep="first")]
    sentiment_df["Sentiment"] = pd.to_numeric(
        sentiment_df["Sentiment"], errors="coerce"
    )

    # Create full date range
    full_date_range = pd.date_range(
        start=min(stock_df.index.min(), sentiment_df.index.min()),
        end=max(stock_df.index.max(), sentiment_df.index.max()),
        freq="B",  # Business days only
    )

    # Reindex and forward fill
    stock_df = stock_df.reindex(full_date_range)
    sentiment_df = sentiment_df.reindex(full_date_range)
    stock_df = stock_df.ffill()

    # Create feature dataset
    data = pd.DataFrame(
        {
            "Close": stock_df["Close"],
            "Volume": stock_df["Volume"],
            "SMA_5": stock_df["SMA_5"],
            "SMA_20": stock_df["SMA_20"],
            "RSI": stock_df["RSI"],
            "Volatility": stock_df["Volatility"],
            "Sentiment": sentiment_df["Sentiment"],
        }
    )

    # Calculate additional features with shorter windows
    data["Price_Momentum"] = data["Close"].pct_change(2)  # Reduced from 5 to 2
    data["Volume_Change"] = data["Volume"].pct_change()
    data["SMA_Cross"] = (data["SMA_5"] > data["SMA_20"]).astype(int)

    # Add sentiment features with shorter window
    data["Sentiment_MA"] = data["Sentiment"].rolling(window=window).mean()
    data["Sentiment_Std"] = data["Sentiment"].rolling(window=window).std()

    # Remove NaN values
    data = data.dropna()

    return data


def train_and_predict(data, prediction_window=1):
    """Train model with cross-validation and confidence intervals."""
    # Prepare features
    feature_cols = [
        "SMA_5",
        "SMA_20",
        "RSI",
        "Volatility",
        "Sentiment_MA",
        "Sentiment_Std",
        "Price_Momentum",
        "Volume_Change",
        "SMA_Cross",
    ]
    X = data[feature_cols]
    y = data["Close"].shift(-prediction_window).dropna()

    # Align X with shifted y
    X = X.iloc[:-prediction_window]

    # Initialize models
    model = Ridge(alpha=1.0)  # Use Ridge regression for better stability

    # Modified time series split for small dataset
    n_samples = len(X)
    if n_samples <= 3:
        # For very small datasets, use a simple train-test split
        train_size = max(1, n_samples - 1)  # Leave at least 1 sample for testing
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        cv_scores = [mean_absolute_percentage_error(y_test, pred)]
    else:
        # For larger datasets, use TimeSeriesSplit with appropriate number of splits
        n_splits = min(2, n_samples - 1)  # Use at most n_samples-1 splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        predictions = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions.extend(pred)
            cv_scores.append(mean_absolute_percentage_error(y_test, pred))

    # Final prediction
    model.fit(X, y)  # Fit on all data
    last_features = X.iloc[-1:]
    prediction = model.predict(last_features)[0]

    # Calculate prediction interval
    confidence_level = 0.95
    cv_std = np.std(cv_scores) if len(cv_scores) > 1 else np.mean(cv_scores) * 0.1
    margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * cv_std

    prediction_interval = (
        prediction * (1 - margin_of_error),
        prediction * (1 + margin_of_error),
    )

    return prediction, prediction_interval, np.mean(cv_scores)


def plot_results(data, prediction, prediction_interval):
    """Enhanced visualization with prediction intervals."""
    fig = go.Figure()

    # Plot actual prices
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Actual Price",
            line=dict(color="blue"),
        )
    )

    # Plot prediction with interval
    future_date = data.index[-1] + timedelta(days=1)
    fig.add_trace(
        go.Scatter(
            x=[future_date],
            y=[prediction],
            mode="markers",
            name="Predicted Price",
            marker=dict(color="red", size=10),
        )
    )

    # Add prediction interval
    fig.add_trace(
        go.Scatter(
            x=[future_date, future_date],
            y=[prediction_interval[0], prediction_interval[1]],
            mode="lines",
            name="95% Prediction Interval",
            line=dict(color="rgba(255,0,0,0.2)", width=2),
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Stock Price Prediction with Confidence Interval",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        hovermode="x unified",
    )

    fig.show()


def main():
    # Updated date range
    target_date = "2024-11-22"  # Last trading day we want to predict
    end_date = "2024-11-21"  # Day before target date
    start_date = "2024-11-04"  # Extended start date
    company = "NOC"

    try:
        # Fetch data with extended date range
        print(f"Fetching sentiment data from {start_date} to {target_date}...")
        sentiment_data = get_sentiment_scores(company, start_date, target_date)

        print(f"Fetching stock data...")
        stock_data = fetch_stock_data(company, start_date, target_date)

        # Prepare features
        data = prepare_features(
            stock_data, sentiment_data, window=3
        )  # Restored window to 3 since we have more data

        if len(data) < 5:  # Restored original minimum required days
            raise ValueError(f"Insufficient data points. Got {len(data)} days.")

        # Make prediction
        prediction, prediction_interval, cv_mape = train_and_predict(data)

        # Get actual closing price for comparison
        actual_price = (
            yf.Ticker(company)
            .history(
                start=target_date,
                end=(
                    datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d"),
            )["Close"]
            .iloc[0]
            if not datetime.strptime(target_date, "%Y-%m-%d").date()
            > datetime.now().date()
            else None
        )

        # Print results
        print(f"\nPrediction Summary for {target_date}:")
        print(f"Last Known Close Price (11/21): ${data['Close'].iloc[-1]:.2f}")
        print(f"Predicted Price: ${prediction:.2f}")
        print(
            f"95% Prediction Interval: ${prediction_interval[0]:.2f} to ${prediction_interval[1]:.2f}"
        )
        print(f"Model MAPE: {cv_mape:.2f}%")

        if actual_price is not None:
            prediction_error = abs(prediction - actual_price) / actual_price * 100
            print(f"\nActual Close Price: ${actual_price:.2f}")
            print(f"Prediction Error: {prediction_error:.2f}%")
            print(
                f"Within Prediction Interval: {prediction_interval[0] <= actual_price <= prediction_interval[1]}"
            )

        # Create enhanced visualization
        fig = go.Figure()

        # Plot historical prices
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Historical Price",
                line=dict(color="blue"),
            )
        )

        # Plot sentiment overlay
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Sentiment_MA"]
                * data["Close"].mean(),  # Scale sentiment to price range
                mode="lines",
                name="Sentiment Trend",
                line=dict(color="purple", dash="dash"),
                opacity=0.5,
                yaxis="y2",
            )
        )

        # Add prediction point
        prediction_date = datetime.strptime(target_date, "%Y-%m-%d")
        fig.add_trace(
            go.Scatter(
                x=[prediction_date],
                y=[prediction],
                mode="markers",
                name="Predicted Price",
                marker=dict(color="red", size=10),
            )
        )

        # Add actual price if available
        if actual_price is not None:
            fig.add_trace(
                go.Scatter(
                    x=[prediction_date],
                    y=[actual_price],
                    mode="markers",
                    name="Actual Price",
                    marker=dict(color="green", size=10),
                )
            )

        # Add prediction interval
        fig.add_trace(
            go.Scatter(
                x=[prediction_date, prediction_date],
                y=[prediction_interval[0], prediction_interval[1]],
                mode="lines",
                name="95% Prediction Interval",
                line=dict(color="rgba(255,0,0,0.2)", width=2),
            )
        )

        # Update layout with dual y-axis
        fig.update_layout(
            title=f"NOC Stock Price Prediction vs Actual for {target_date}",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            yaxis2=dict(
                title="Sentiment Trend", overlaying="y", side="right", showgrid=False
            ),
            template="plotly_white",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        fig.show()

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
