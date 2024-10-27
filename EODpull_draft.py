import requests
import pandas as pd
import re
from datetime import datetime
from secret import EODHD_key  # Ensure this module exists and contains your API key

# Define your EOD Historical Data API key
api_key = EODHD_key  # Replace with your actual API key

# Define the company symbols (Raytheon, Honeywell, Lockheed Martin)
symbols = {
    "RTX.US": "Raytheon",
    "HON.US": "Honeywell",
    "LMT.US": "Lockheed Martin",
}

# Get today's date in the format Y-m-d
today = datetime.utcnow().strftime("%Y-%m-%d")
test_data = "2024-10-15"  # Example for test data

# Construct the URL for Financial News Feed API
url = "https://eodhistoricaldata.com/api/news"

# Function to make an API call for each symbol
def fetch_news_data(symbol):
    params = {
        "s": symbol,
        "api_token": api_key,
        "from": test_data,  # Using test_data for this example
        "to": today,
        "limit": 100,  # Limit to 100 articles (you can change this limit)
        "fmt": "json"
    }
    # Make the request
    response = requests.get(url, params=params)
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()  # Return the JSON data
    else:
        print(f"Error fetching data for {symbol}: {response.status_code} - {response.text}")
        return None

# Fetch data for each symbol
dataRTX = fetch_news_data("RTX.US")
dataHON = fetch_news_data("HON.US")
dataLMT = fetch_news_data("LMT.US")

# Print the fetched data
print("Raytheon (RTX) News Data:", dataRTX)
print("Honeywell (HON) News Data:", dataHON)
print("Lockheed Martin (LMT) News Data:", dataLMT)


# import requests

# url = f'https://eodhd.com/api/news?s=AAPL.US&offset=0&limit=10&api_token= 6716eb33591136.32003731&fmt=json'
# data = requests.get(url).json()

# print(data)



# # Function to clean HTML elements and unwanted characters
# def clean_text(text):
#     # Remove HTML tags using a regex
#     text = re.sub(r"<[^>]+>", "", text)
#     # Replace multiple newlines and extra spaces with a single space
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# # Function to check if a company symbol is mentioned in the title or description
# def company_mentioned(text, symbols):
#     for symbol, company in symbols.items():
#         if symbol in text or company in text:
#             return symbol  # Return the symbol if mentioned
#     return None  # Return None if no symbol is mentioned

# # Make the request to the EODHD API
# response = requests.get(url, params=params)

# # Check if the request was successful
# if response.status_code == 200:
#     news_data = response.json()

#     # Create dictionaries to store highlights and sentiment scores for each company
#     company_data = {symbol: [] for symbol in symbols}
#     company_sentiment_scores = {symbol: [] for symbol in symbols}

#     # Iterate through the articles
#     for article in news_data:
#         title = clean_text(article.get("title", ""))
#         description = clean_text(article.get("content", ""))
#         sentiment_score = article.get("sentiment", 0)  # Sentiment score may vary, so use 0 if not present
#         publication_date = article.get("date", "")

#         # Check if any company is mentioned in the title or description
#         symbol_mentioned = company_mentioned(title + description, symbols)
#         if symbol_mentioned:
#             # Add the article details to the corresponding company's list
#             company_data[symbol_mentioned].append(
#                 {
#                     "Title": title,
#                     "Description": description,
#                     "Publication Date": publication_date,
#                     "Sentiment Score": sentiment_score,
#                 }
#             )

#             # Collect non-zero sentiment scores for averaging
#             try:
#                 score = float(sentiment_score)
#                 if score != 0:  # Only consider non-zero scores
#                     company_sentiment_scores[symbol_mentioned].append(score)
#             except ValueError:
#                 continue  # Skip invalid scores

#     # Write each company's data to a separate CSV using Pandas
#     for symbol, data in company_data.items():
#         if data:  # Check if there are any entries for the company
#             df = pd.DataFrame(data)
#             company_name = symbols[symbol]
#             df.to_csv(
#                 f"{company_name}_news_highlights.csv", index=False, encoding="utf-8"
#             )

#     # Calculate and print average sentiment score for each company
#     for symbol, scores in company_sentiment_scores.items():
#         if scores:  # If there are valid sentiment scores
#             average_sentiment = sum(scores) / len(scores)
#             company_name = symbols[symbol]
#             print(
#                 f"Average sentiment for {company_name} ({symbol}): {average_sentiment:.2f}"
#             )

# else:
#     print(f"Error: {response.status_code}, {response.text}")
