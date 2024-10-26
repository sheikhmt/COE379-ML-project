import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import scipy.special
import yfinance as yf
from secret import FINHUB_KEY
import finnhub
from scipy import stats

# Initialize Finnhub client (for news only)
finnhub_client = finnhub.Client(api_key=FINHUB_KEY)

# Fetch the company news for RTX
news_data = finnhub_client.company_news("RTX", _from="2024-10-01", to="2024-10-21")

# Extract summaries
summaries = [news_item["summary"] for news_item in news_data]

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


# Function to get sentiment scores
def get_sentiment_scores(summaries):
    preds_proba = []
    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}

    for summary in summaries:
        with torch.no_grad():
            input_sequence = tokenizer(summary, return_tensors="pt", **tokenizer_kwargs)
            logits = model_finbert(**input_sequence).logits
            scores = {
                k: v
                for k, v in zip(
                    model_finbert.config.id2label.values(),
                    scipy.special.softmax(logits.numpy().squeeze()),
                )
            }
            probabilityFinbert = max(scores.values())
            preds_proba.append(probabilityFinbert)

    return preds_proba


# Get sentiment scores
sentiment_scores = get_sentiment_scores(summaries)

# Remove outliers from sentiment scores using Z-score
sentiment_scores = np.array(sentiment_scores)
z_scores = np.abs(stats.zscore(sentiment_scores))
sentiment_scores = sentiment_scores[
    z_scores < 2
]  # Keep data within 2 standard deviations

# Fetch Raytheon stock data using yfinance
stock_df = yf.download("RTX", start="2024-10-01", end="2024-10-21", interval="1d")

# Ensure stock_df and sentiment_df are timezone-naive
stock_df.index = stock_df.index.tz_localize(None)

# Create a DataFrame for sentiment analysis
sentiment_df = pd.DataFrame(
    {
        "Date": pd.date_range(start="2024-10-01", periods=len(sentiment_scores)),
        "Sentiment": sentiment_scores,
    }
)
sentiment_df.set_index("Date", inplace=True)
sentiment_df.index = sentiment_df.index.tz_localize(None)

# Filter sentiment_df to match stock_df date range and fill missing dates
sentiment_df = sentiment_df.reindex(stock_df.index).interpolate()

# Apply a rolling average to smooth the data
window = 3  # Rolling window size (can be adjusted)
stock_df["Close_Smoothed"] = stock_df["Close"].rolling(window=window).mean()
sentiment_df["Sentiment_Smoothed"] = (
    sentiment_df["Sentiment"].rolling(window=window).mean()
)

# Align and create the combined DataFrame
data = pd.DataFrame(
    {
        "Close": stock_df["Close_Smoothed"],
        "Sentiment": sentiment_df["Sentiment_Smoothed"],
    },
    index=stock_df.index,
).dropna()  # Drop NaN values due to rolling mean

# Calculate correlations
pearson_corr, p_value = stats.pearsonr(data["Close"], data["Sentiment"])

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Close Price on primary y-axis
color = "tab:blue"
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price (USD)", color=color)
line1 = ax1.plot(data.index, data["Close"], color=color, label="Stock Price (Smoothed)")
ax1.tick_params(axis="y", labelcolor=color)

# Create secondary y-axis for Sentiment
ax2 = ax1.twinx()
color = "tab:orange"
ax2.set_ylabel("Sentiment Score", color=color)
line2 = ax2.plot(
    data.index, data["Sentiment"], color=color, label="Sentiment (Smoothed)"
)
ax2.tick_params(axis="y", labelcolor=color)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add correlation information
plt.title(
    f"RTX Stock Price and Smoothed Sentiment Over Time\nPearson Correlation: {pearson_corr:.3f} (p-value: {p_value:.3f})"
)

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.show()

# Print additional statistics
print("\nCorrelation Analysis:")
print(f"Pearson correlation coefficient: {pearson_corr:.3f}")
print(f"P-value: {p_value:.3f}")

# Calculate daily changes
data["Price_Change"] = data["Close"].pct_change()
data["Sentiment_Change"] = data["Sentiment"].pct_change()

# Calculate correlation for daily changes
daily_corr = data["Price_Change"].corr(data["Sentiment_Change"])
print("\nDaily Changes Correlation:")
print(f"Correlation between daily changes: {daily_corr:.3f}")
