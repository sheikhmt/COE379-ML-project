import yfinance as yf
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Step 1: Stock Price Data Collection for 2021-2023
# Define stock symbols (Lockheed Martin, Honeywell, Raytheon)
stocks = ['LMT', 'HON', 'RTX']

# Download historical stock price data from 2021 to 2023
stock_data = yf.download(stocks, start="2021-01-01", end="2023-12-31")

# Step 2: Feature Engineering (Moving Averages, Returns)
# Calculate 50-day moving average for each stock
stock_data['LMT_50MA'] = stock_data['Close']['LMT'].rolling(window=50).mean()
stock_data['HON_50MA'] = stock_data['Close']['HON'].rolling(window=50).mean()
stock_data['RTX_50MA'] = stock_data['Close']['RTX'].rolling(window=50).mean()

# Calculate daily returns (percentage change)
stock_data['LMT_Returns'] = stock_data['Close']['LMT'].pct_change() * 100
stock_data['HON_Returns'] = stock_data['Close']['HON'].pct_change() * 100
stock_data['RTX_Returns'] = stock_data['Close']['RTX'].pct_change() * 100

# Step 3: Sentiment Analysis on Annual Reports
# Example text from annual reports (this would normally come from report parsing)
annual_report_texts = {
    'LMT': "Lockheed Martin had an exceptional year with record profits and expansion in new markets...",
    'HON': "Honeywell experienced growth across all segments, but faced challenges in supply chains...",
    'RTX': "Raytheon reported stable revenue, focusing on innovations in defense technologies..."
}

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment for each company
sentiment_scores = {}
for company, text in annual_report_texts.items():
    sentiment_scores[company] = analyzer.polarity_scores(text)

# Convert sentiment scores to a DataFrame for merging
sentiment_df = pd.DataFrame(sentiment_scores).T  # Transpose to align with stock_data

# Step 4: Combine Stock Data with Sentiment Data
# Merge the stock data with sentiment scores (this assumes one sentiment score for the entire period, more complex models can use daily news sentiment)
combined_data = pd.merge(stock_data, sentiment_df, left_index=True, right_index=True, how='inner')

# Step 5: LSTM Model for Price Prediction (using scaled data)
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

# Step 6: LSTM Model Definition
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(1)  # Output layer (predict stock price)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64)

# Step 7: Random Forest for Direction Prediction (Optional)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create labels: 1 if the price increases, 0 if it decreases
combined_data['Price_Up'] = (combined_data['Close']['LMT'].shift(-1) > combined_data['Close']['LMT']).astype(int)

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

# Step 8: Save Data for Later Use
combined_data.to_csv('defense_stock_with_sentiment.csv')
