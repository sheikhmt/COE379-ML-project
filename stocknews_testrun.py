import requests
import json
from datetime import datetime
from secret import stock_news_api_key  # Import your Stock News API key

# Stock News API settings
stock_news_url = "https://stocknewsapi.com/api/v1"

# Function to format the date to YYYY-MM-DD
def format_date(date_str):
    """
    Convert date string from 'Mon, 02 Dec 2024 18:07:47 -0500' to 'YYYY-MM-DD'.
    :param date_str: Original date string
    :return: Formatted date string in 'YYYY-MM-DD'
    """
    try:
        return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z").strftime("%Y-%m-%d")
    except ValueError:
        print(f"Error parsing date: {date_str}")
        return ""

# Function to fetch news from Stock News API
def fetch_stock_news(tickers, items, page):
    """
    Fetch news for given tickers from Stock News API.
    :param tickers: Comma-separated ticker symbols (e.g., 'RTX,NOC,LMT')
    :param items: Number of articles to fetch per page
    :param page: Page number for pagination
    :return: JSON response containing news data
    """
    params = {
        "tickers": tickers,  # Comma-separated ticker symbols
        "items": items,      # Number of articles to fetch
        "page": page,        # Page number for pagination
        "token": stock_news_api_key,  # Stock News API key
    }
    response = requests.get(stock_news_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Stock News API error for {tickers}: {response.status_code} - {response.text}")
        return []

# Function to save news data to JSON
def save_to_json(data, company_name):
    """
    Save the news data to a JSON file.
    :param data: News data to save
    :param company_name: Name of the company for the file name
    """
    if data:
        # Format file name as company_name-MM-DD-YYYY.json
        date_str = datetime.now().strftime("%m-%d-%Y")
        file_name = f"{company_name.lower()}-{date_str}.json"
        with open(file_name, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Saved news data for {company_name} to {file_name}")
    else:
        print(f"No news data found for {company_name}.")

# Main script
if __name__ == "__main__":
    # Define companies and their ticker symbols
    tickers = {
        "RTX": "Raytheon",
        "NOC": "Northrop",
        "LMT": "Lockheed Martin",
    }

    # Fetch and save news for each ticker
    for ticker, company_name in tickers.items():
        print(f"Fetching Stock News API news for {company_name} ({ticker})...")
        page = 1
        items = 3

        stock_news_data = fetch_stock_news(ticker, items, page)

        # Extract relevant fields and format the data
        filtered_news = []
        if "data" in stock_news_data:
            for article in stock_news_data["data"]:
                formatted_date = format_date(article.get("date", ""))
                if formatted_date:
                    filtered_news.append(
                        {
                            "Date": formatted_date,
                            "Title": article.get("title", ""),
                            "Summary": article.get("text", ""),
                            "URL": article.get("news_url", ""),
                            "Source": "Stock News API",
                        }
                    )

        # Save the filtered news data to JSON
        save_to_json(filtered_news, company_name)
