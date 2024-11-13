import os
import requests
import json
from datetime import datetime, timedelta
from secret import EODHD_key
import re

# Define your EOD Historical Data API key
api_key = EODHD_key

# Define the company symbols (Raytheon, Honeywell, Lockheed Martin)
symbols = {
    "RTX.US": "Raytheon",
    "HON.US": "Honeywell",
    "LMT.US": "Lockheed Martin",
}

# Define the API URL
url = "https://eodhd.com/api/news"


# Function to get articles data for a specific date range and save articles from the same day into the same file
def get_and_save_articles(symbol, company_name, start_date, end_date):
    # Define parameters for API call
    params = {
        "s": symbol,
        "api_token": api_key,
        "from": start_date,
        "to": end_date,
        "limit": 1000,
        "fmt": "json",
    }

    # Make the API request
    response = requests.get(url, params=params)
    data = response.json()

    # Define directory path based on the company name and create it if it doesn't exist
    directory_path = os.path.join(company_name, "6_Months")
    os.makedirs(directory_path, exist_ok=True)

    # Dictionary to hold articles by date
    articles_by_date = {}

    # Organize articles by date
    for article in data:
        article_date = article.get("date", "unknown").split("T")[
            0
        ]  # Get the date portion
        if article_date not in articles_by_date:
            articles_by_date[article_date] = []
        articles_by_date[article_date].append(article)

    # Save articles for each date into a single JSON file
    for article_date, articles in articles_by_date.items():
        file_name = f"{article_date}.json"  # Use the date as the filename
        file_path = os.path.join(directory_path, file_name)

        # Save the articles data as a JSON file
        with open(file_path, "w") as json_file:
            json.dump(articles, json_file, indent=4)

    print(f"Saved articles for {company_name} in {directory_path}")


# Main function to fetch and save articles for each company
def fetch_and_save_all_articles():
    # Define the end and start date for the last 6 months
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=6 * 30)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Fetch and save articles for each company
    for symbol, company_name in symbols.items():
        get_and_save_articles(symbol, company_name, start_date_str, end_date_str)


if __name__ == "__main__":
    fetch_and_save_all_articles()
