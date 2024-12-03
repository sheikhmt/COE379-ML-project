import requests
import json
from datetime import datetime, timedelta
from secret import alpha_vantage_key  # Import your Alpha Vantage API key

# Alpha Vantage API settings
alpha_vantage_url = "https://www.alphavantage.co/query"

# Function to format the date to YYYY-MM-DD
def format_date(date_str):
    """
    Convert date string from 'YYYYMMDDTHHMM' to 'YYYY-MM-DD'.
    :param date_str: Original date string
    :return: Formatted date string in 'YYYY-MM-DD'
    """
    try:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    except IndexError:
        print(f"Error parsing date: {date_str}")
        return ""

# Function to fetch news from Alpha Vantage API
def fetch_alpha_vantage_news(symbol):
    """
    Fetch news articles for a specific ticker symbol from Alpha Vantage API.
    :param symbol: Ticker symbol (e.g., 'RTX')
    :return: List of news articles
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "apikey": alpha_vantage_key,
    }
    response = requests.get(alpha_vantage_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Alpha Vantage API error for {symbol}: {response.status_code} - {response.text}")
        return []

# Function to save grouped news data by date into JSON files
def save_grouped_news(data, company_name):
    """
    Save news articles grouped by date into separate JSON files.
    :param data: List of news articles
    :param company_name: Name of the company
    """
    grouped_data = {}
    for article in data:
        date = article.get("Date", "")
        if date:
            if date not in grouped_data:
                grouped_data[date] = []
            grouped_data[date].append(article)

    for date, articles in grouped_data.items():
        file_name = f"{company_name.lower()}-{date}.json"
        with open(file_name, "w", encoding="utf-8") as json_file:
            json.dump(articles, json_file, indent=4)
        print(f"Saved news data for {company_name} on {date} to {file_name}")

# Main function to fetch and process news for a date range
def process_news_for_date_range(tickers, start_date, end_date):
    """
    Fetch and process news articles for a given date range.
    :param tickers: Dictionary of ticker symbols and company names
    :param start_date: Start date in YYYY-MM-DD format
    :param end_date: End date in YYYY-MM-DD format
    """
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Processing news for {date_str}...")

        for symbol, company_name in tickers.items():
            print(f"Fetching Alpha Vantage news for {company_name} ({symbol})...")
            alpha_vantage_data = fetch_alpha_vantage_news(symbol)

            # Extract relevant fields and group by date
            filtered_news = []
            if "feed" in alpha_vantage_data:
                for article in alpha_vantage_data["feed"]:
                    formatted_date = format_date(article.get("time_published", ""))
                    if formatted_date == date_str:
                        filtered_news.append(
                            {
                                "Date": formatted_date,
                                "Title": article.get("title", ""),
                                "Summary": article.get("summary", ""),
                                "URL": article.get("url", ""),
                                "Source": "Alpha Vantage API",
                            }
                        )

            # Save grouped news data to JSON
            save_grouped_news(filtered_news, company_name)

        current_date += timedelta(days=1)

# Main script
if __name__ == "__main__":
    # Define company tickers and names
    tickers = {
        "RTX": "Raytheon",
        "NOC": "Northrop",
        "LMT": "Lockheed Martin",
    }

    # Define date range
    start_date = "2024-12-01"  # Start date in YYYY-MM-DD format
    end_date = "2024-12-03"    # End date in YYYY-MM-DD format

    # Process news for the given date range
    process_news_for_date_range(tickers, start_date, end_date)
