import yfinance as yf
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# # Step 1: Stock Price Data Collection for 2021-2023
# def get_stock_data():
#     # Define stock symbols (Lockheed Martin, Honeywell, Raytheon)
#     stocks = ['LMT', 'HON', 'RTX']

#     # Download historical stock price data from 2021 to 2023
#     stock_data = yf.download(stocks, start="2021-01-01", end="2023-12-31")

#     # Calculate 50-day moving average for each stock
#     stock_data['LMT_50MA'] = stock_data['Close']['LMT'].rolling(window=50).mean()
#     stock_data['HON_50MA'] = stock_data['Close']['HON'].rolling(window=50).mean()
#     stock_data['RTX_50MA'] = stock_data['Close']['RTX'].rolling(window=50).mean()

#     # Calculate daily returns (percentage change)
#     stock_data['LMT_Returns'] = stock_data['Close']['LMT'].pct_change() * 100
#     stock_data['HON_Returns'] = stock_data['Close']['HON'].pct_change() * 100
#     stock_data['RTX_Returns'] = stock_data['Close']['RTX'].pct_change() * 100

#     # Reset index for easier merging later
#     stock_data = stock_data.reset_index()
#     return stock_data

# Step 2: Sentiment Data Collection using MarketAux API
def get_sentiment_data(api_key, companies=['LMT', 'HON', 'RTX']):
    sentiment_data = []

    for company in companies:
        url = f"https://api.marketaux.com/v1/news/all?symbols={company}&filter_entities=true&api_token={api_key}"
        response = requests.get(url).json()

        # Parse news articles and extract sentiment
        for article in response['data']:
            sentiment = article['sentiment']
            date = article['published_at'][:10]  # Format date
            sentiment_score = 1 if sentiment == 'positive' else (-1 if sentiment == 'negative' else 0)
            sentiment_data.append({'company': company, 'date': date, 'sentiment_score': sentiment_score})

    # Convert to DataFrame
    sentiment_df = pd.DataFrame(sentiment_data)
    return sentiment_df

# Step 3: Combine Stock and Sentiment Data
def combine_data(stock_data, sentiment_df):
    # Extract date and stock ticker columns from stock_data for merging
    stock_data['date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')

    # Add a company column to match the sentiment data format
    stock_data_lmt = stock_data[['date', 'Close', 'LMT_50MA', 'LMT_Returns']].copy()
    stock_data_lmt['company'] = 'LMT'

    stock_data_hon = stock_data[['date', 'Close', 'HON_50MA', 'HON_Returns']].copy()
    stock_data_hon['company'] = 'HON'

    stock_data_rtx = stock_data[['date', 'Close', 'RTX_50MA', 'RTX_Returns']].copy()
    stock_data_rtx['company'] = 'RTX'

    # Combine stock data from all three companies
    combined_stock_data = pd.concat([stock_data_lmt, stock_data_hon, stock_data_rtx])

    # Merge stock data with sentiment data
    combined_data = pd.merge(combined_stock_data, sentiment_df, on=['date', 'company'], how='inner')
    return combined_data

# Step 4: LSTM Model for Price Prediction
def build_lstm_model(combined_data):
    # Prepare the data for LSTM (Scale the data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_data.dropna())  # Drop NA values for simplicity

    # Create input-output sequences (X, y)
    X_train, y_train = [], []
    window_size = 130  # Use the last 130 days of data to predict the next day
    for i in range(window_size, len(scaled_data)):
        X_train.append(scaled_data[i-window_size:i, :])  # Input sequence
        y_train.append(scaled_data[i, 0])  # Target (next day's closing price)

    X_train, y_train = np.array(X_train), np.array(y_train)

    # LSTM Model Definition
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(1)  # Output layer (predict stock price)
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=64)
    return model

# Step 5: Random Forest for Stock Direction Prediction
def build_random_forest_model(combined_data):
    # Create labels: 1 if the price increases, 0 if it decreases
    combined_data['Price_Up'] = (combined_data['Close'].shift(-1) > combined_data['Close']).astype(int)

    # Define the features (X) and labels (y)
    X = combined_data.drop(columns=['Price_Up'])
    y = combined_data['Price_Up']

    # Split the data
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_rf, y_train_rf)

    # Evaluate the model's accuracy
    accuracy = rf_model.score(X_test_rf, y_test_rf)
    print(f"Random Forest Accuracy: {accuracy}")
    return rf_model

# Example Usage
if __name__ == "__main__":
    api_key = "your_marketaux_api_key"  # Replace with your MarketAux API key
#API key: OwqFK6zS6zvEkEjyrRRguEOkK0dbyJZn3vS2O9Vq
    # Step 1: Get stock price data
    stock_data = get_stock_data()

    # Step 2: Get sentiment data from MarketAux API
    sentiment_df = get_sentiment_data(api_key)

    # Step 3: Combine stock and sentiment data
    combined_data = combine_data(stock_data, sentiment_df)

    # Step 4: Train LSTM model for price prediction
    lstm_model = build_lstm_model(combined_data)

    # Step 5: Train Random Forest model for direction prediction
    rf_model = build_random_forest_model(combined_data)
    
    # Save combined data to CSV for future use
    combined_data.to_csv('defense_stock_with_sentiment.csv')
