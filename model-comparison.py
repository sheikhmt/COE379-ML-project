import json
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google.cloud import storage
import yfinance as yf
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
import scipy
from scipy import stats
from datetime import datetime, timedelta
from google.oauth2 import service_account
import warnings

warnings.filterwarnings("ignore")

# [Previous imports and configurations remain the same...]
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


def evaluate_models(X_train, X_test, y_train, y_test):
    """Evaluate multiple models and return their performance metrics."""
    models = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "SVR": SVR(kernel="rbf"),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42
        ),
    }

    results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                "model": model,
                "mse": mse,
                "mape": mape,
                "r2": r2,
                "scaler": scaler,
            }

        except Exception as e:
            print(f"Error with {name}: {str(e)}")

    return results


def plot_model_comparison(results):
    """Plot model comparison results."""
    fig = go.Figure()

    models = list(results.keys())
    mape_scores = [results[model]["mape"] * 100 for model in models]
    r2_scores = [results[model]["r2"] for model in models]

    # Add MAPE bars
    fig.add_trace(
        go.Bar(name="MAPE (%)", x=models, y=mape_scores, marker_color="indianred")
    )

    # Add R² bars
    fig.add_trace(
        go.Bar(name="R² Score", x=models, y=r2_scores, marker_color="lightseagreen")
    )

    fig.update_layout(
        title="Model Performance Comparison",
        barmode="group",
        xaxis_title="Models",
        yaxis_title="Score",
        template="plotly_white",
    )

    return fig


def train_and_predict(data, prediction_window=1):
    """Enhanced training and prediction with model comparison."""
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

    # Split the data temporally
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Evaluate all models
    results = evaluate_models(X_train, X_test, y_train, y_test)

    # Select best model based on MAPE
    best_model_name = min(results.keys(), key=lambda k: results[k]["mape"])
    best_model_info = results[best_model_name]

    print(f"\nModel Performance Summary:")
    for name, info in results.items():
        print(f"{name}:")
        print(f"  MAPE: {info['mape']*100:.2f}%")
        print(f"  R2 Score: {info['r2']:.4f}")
    print(f"\nSelected Model: {best_model_name}")

    # Create and display model comparison plot
    comparison_fig = plot_model_comparison(results)
    comparison_fig.show()

    # Prepare final prediction with best model
    scaler = best_model_info["scaler"]
    model = best_model_info["model"]

    # Scale the entire dataset and retrain on all data
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    # Prepare last features for prediction
    last_features = X.iloc[-1:].copy()
    last_features_scaled = scaler.transform(last_features)

    # Make prediction
    prediction = model.predict(last_features_scaled)[0]

    # Calculate prediction interval using bootstrap
    n_iterations = 100
    bootstrap_predictions = []
    for _ in range(n_iterations):
        # Sample with replacement
        idx = np.random.randint(0, len(X), size=len(X))
        X_boot = X_scaled[idx]
        y_boot = y.iloc[idx]

        # Fit model and predict
        model.fit(X_boot, y_boot)
        bootstrap_predictions.append(model.predict(last_features_scaled)[0])

    # Calculate prediction interval
    confidence_level = 0.95
    interval = np.percentile(
        bootstrap_predictions,
        [100 * (1 - confidence_level) / 2, 100 * (1 + confidence_level) / 2],
    )

    return prediction, interval, best_model_info["mape"], best_model_name


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
        # prediction, prediction_interval, cv_mape = train_and_predict(data)

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

        # Make prediction with model comparison
        prediction, prediction_interval, cv_mape, best_model = train_and_predict(data)

        # Print results
        print(f"\nPrediction Summary for {target_date}:")
        print(f"Best Model: {best_model}")
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
