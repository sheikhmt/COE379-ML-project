import requests
import re
import pandas as pd
from datetime import datetime
from secrets_1 import MARKET_API

# Set your API token here
api_token = MARKET_API

# Define the company symbols (Raytheon, Honeywell, Lockheed Martin)
symbols = {
    "RTX": "Raytheon",
    "HON": "Honeywell",
    "LMT": "Lockheed Martin",
}  # Map symbols to company names

# Get today's date in the format Y-m-d
today = datetime.utcnow().strftime("%Y-%m-%d")
test_data = "2024-10-15"

# Construct the URL
url = f"https://api.marketaux.com/v1/news/all"
url = f"https://eodhd.com/api/news"

params={
    "s":"RTX.US,"
}

# Define parameters for the API call
params = {
    "symbols": ",".join(
        symbols.keys()
    ),  # Convert the list of symbols to a comma-separated string
    "published_on": test_data,  # Using test_data for this example
    "language": "en",
    "api_token": api_token,
}


# Function to clean HTML elements and unwanted characters
def clean_text(text):
    # Remove HTML tags using a regex
    text = re.sub(r"<[^>]+>", "", text)
    # Replace multiple newlines and extra spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Function to check if a company symbol is mentioned in the highlight text
def company_mentioned(highlight_text, symbols):
    for symbol, company in symbols.items():
        if symbol in highlight_text or company in highlight_text:
            return symbol  # Return the symbol if mentioned
    return None  # Return None if no symbol is mentioned


# Make the request to the Marketaux API
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    news_data = response.json()

    # Create dictionaries to store highlights and sentiment scores for each company
    company_data = {symbol: [] for symbol in symbols}
    company_sentiment_scores = {symbol: [] for symbol in symbols}

    # Iterate through the articles
    for article in news_data.get("data", []):
        title = clean_text(article["title"])
        url = article["url"]
        description = clean_text(article["description"])

        # Iterate through each entity and its highlights
        for entity in article.get("entities", []):
            for highlight in entity.get("highlights", []):
                highlight_text = clean_text(highlight.get("highlight", ""))
                sentiment_score = highlight.get("sentiment", "")

                # Check if any company is mentioned in the highlight
                symbol_mentioned = company_mentioned(highlight_text, symbols)
                if symbol_mentioned:
                    # Add the article details to the corresponding company's list
                    company_data[symbol_mentioned].append(
                        {
                            "Title": title,
                            "URL": url,
                            "Description": description,
                            "Highlight": highlight_text,
                            "Sentiment Score": sentiment_score,
                        }
                    )

                    # Collect non-zero sentiment scores for averaging
                    try:
                        score = float(sentiment_score)
                        if score != 0:  # Only consider non-zero scores
                            company_sentiment_scores[symbol_mentioned].append(score)
                    except ValueError:
                        continue  # Skip invalid scores

    # Write each company's data to a separate CSV using Pandas
    for symbol, data in company_data.items():
        if data:  # Check if there are any entries for the company
            df = pd.DataFrame(data)
            company_name = symbols[symbol]
            df.to_csv(
                f"{company_name}_news_highlights.csv", index=False, encoding="utf-8"
            )

    # Calculate and print average sentiment score for each company
    for symbol, scores in company_sentiment_scores.items():
        if scores:  # If there are valid sentiment scores
            average_sentiment = sum(scores) / len(scores)
            company_name = symbols[symbol]
            print(
                f"Average sentiment for {company_name} ({symbol}): {average_sentiment:.2f}"
            )

else:
    print(f"Error: {response.status_code}, {response.text}")
