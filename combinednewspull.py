import requests
import json
import datetime
import finnhub  # Finnhub client
from google.cloud import storage
from google.oauth2 import service_account
from secret import (
    EODHD_key,
    layer_key,
    stock_news_api_key,
    benzinga_key,
    alpha_vantage_key,
    finnhub_key,  # Finnhub API key
)

# Google Cloud credentials
credentials_path = r"C:\Users\dhn6u\Downloads\coe379-ml-project-8266ccc1ee98.json"
credentials = service_account.Credentials.from_service_account_file(credentials_path)
storage_client = storage.Client(credentials=credentials)

# Finnhub client
finnhub_client = finnhub.Client(api_key=finnhub_key)

# API settings
layer_url = "https://api.apilayer.com/financelayer/news"
eod_url = "https://eodhistoricaldata.com/api/news"
benzinga_url = "https://api.benzinga.com/api/v2/news"
alpha_vantage_url = "https://www.alphavantage.co/query"
stock_news_url = "https://stocknewsapi.com/api/v1"

COMPANIES = {
    "rtx": "fin_text/raytheon/news_articles/2_week",
    "noc": "fin_text/northrop/news_articles/2_week",
    "lmt": "fin_text/lockheed/news_articles/2_week",
}

SYMBOLS = {
    "RTX.US": ("Raytheon", "fin_text/raytheon/news_articles/2_week"),
    "NOC.US": ("Northrop", "fin_text/northrop/news_articles/2_week"),
    "LMT.US": ("Lockheed Martin", "fin_text/lockheed/news_articles/2_week"),
}

# GCS bucket name
bucket_name = "fin_analysis_data"

# Filter and truncate summaries
def filter_summary(summary):
    if not summary:
        return None
    if len(summary) < 100:
        return None
    if len(summary) > 1500:
        return summary[:1500] + "..."
    return summary

# Convert Unix timestamp to YYYY-MM-DD format
def convert_unix_to_date(unix_timestamp):
    return datetime.datetime.utcfromtimestamp(unix_timestamp).strftime("%Y-%m-%d")

# Fetch Layer API news
def fetch_layer_news(ticker, date):
    headers = {"apikey": layer_key}
    params = {
        "tickers": ticker,
        "date_from": date,
        "date_to": date,
        "limit": 100,
    }
    response = requests.get(layer_url, headers=headers, params=params)
    if response.status_code == 200:
        return [
            {
                "Date": article.get("published_at"),
                "Title": article.get("title"),
                "Summary": filter_summary(article.get("description")),
                "URL": article.get("url"),
                "Source": "Layer API",
            }
            for article in response.json().get("data", []) if article.get("published_at", "").startswith(date)
        ]
    else:
        print(f"Layer API error for {ticker}: {response.status_code}")
        return []

# Fetch EOD API news
def fetch_eod_news(symbol, date):
    params = {
        "s": symbol,
        "api_token": EODHD_key,
        "from": date,
        "to": date,
        "limit": 100,
        "fmt": "json",
    }
    response = requests.get(eod_url, params=params)
    if response.status_code == 200:
        return [
            {
                "Date": article.get("date"),
                "Title": article.get("title"),
                "Summary": filter_summary(article.get("content")),
                "URL": article.get("link"),
                "Source": "EOD API",
            }
            for article in response.json() if article.get("date", "").startswith(date)
        ]
    else:
        print(f"EOD API error for {symbol}: {response.status_code}")
        return []

# Fetch Benzinga API news
def fetch_benzinga_news(ticker, date):
    headers = {"accept": "application/json"}
    params = {
        "token": benzinga_key,
        "tickers": ticker.upper(),
        "date": date,
        "displayOutput": "full",
        "pageSize": 100,
    }
    response = requests.get(benzinga_url, headers=headers, params=params)
    if response.status_code == 200:
        return [
            {
                "Date": article.get("created", "")[:10],
                "Title": article.get("title", ""),
                "Summary": filter_summary(article.get("body", "")),
                "URL": article.get("url", ""),
                "Source": "Benzinga API",
            }
            for article in response.json() if article.get("created", "").startswith(date)
        ]
    else:
        print(f"Benzinga API error for {ticker}: {response.status_code}")
        return []

# Fetch Finnhub API news
def fetch_finnhub_news(ticker, date):
    try:
        news = finnhub_client.company_news(ticker, _from=date, to=date)
        return [
            {
                "Date": convert_unix_to_date(article.get("datetime", 0)),
                "Title": article.get("headline", ""),
                "Summary": filter_summary(article.get("summary", "")),
                "URL": article.get("url", ""),
                "Source": "Finnhub API",
            }
            for article in news if convert_unix_to_date(article.get("datetime", 0)) == date
        ]
    except Exception as e:
        print(f"Finnhub API error for {ticker}: {e}")
        return []

# Fetch Alpha Vantage API news
def fetch_alpha_vantage_news(symbol, date):
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "apikey": alpha_vantage_key,
    }
    response = requests.get(alpha_vantage_url, params=params)
    if response.status_code == 200:
        return [
            {
                "Date": article.get("time_published", ""),
                "Title": article.get("title", ""),
                "Summary": filter_summary(article.get("summary", "")),
                "URL": article.get("url", ""),
                "Source": "Alpha Vantage API",
            }
            for article in response.json().get("feed", []) if article.get("time_published", "").startswith(date)
        ]
    else:
        print(f"Alpha Vantage API error for {symbol}: {response.status_code}")
        return []

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
        "type": "Article",
    }
    response = requests.get(stock_news_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Stock News API error for {tickers}: {response.status_code} - {response.text}")
        return []
# # Fetch Stock News API news
# def fetch_stock_news(ticker, date):
#     params = {
#         "tickers": ticker.upper(),
#         "items": 10,
#         "page": 1,
#         "token": stock_news_api_key,
#     }
#     response = requests.get(stock_news_url, params=params)
#     if response.status_code == 200:
#         return [
#             {
#                 "Date": article.get("date", ""),
#                 "Title": article.get("title", ""),
#                 "Summary": filter_summary(article.get("text", "")),
#                 "URL": article.get("news_url", ""),
#                 "Source": "Stock News API",
#             }
#             for article in response.json().get("data", []) if article.get("date", "").startswith(date)
#         ]
#     else:
#         print(f"Stock News API error for {ticker}: {response.status_code}")
#         return []

# Save news articles grouped by date to GCS
def save_to_gcs_grouped_by_date(data, company_name, gcs_folder):
    grouped_data = {}
    for article in data:
        date = article.get("Date", "")[:10]  # Group by YYYY-MM-DD
        if date:
            grouped_data.setdefault(date, []).append(article)

    for date, articles in grouped_data.items():
        file_name = f"{company_name.lower()}-{date}.json"
        json_data = json.dumps(articles, indent=4)
        gcs_path = f"{gcs_folder}/{file_name}"

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(json_data, content_type="application/json")
        print(f"Uploaded {file_name} to gs://{bucket_name}/{gcs_path}")

# Main function to process and upload news
def process_news(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        date = current_date.strftime("%Y-%m-%d")
        print(f"Processing news for {date}...")
        for ticker, gcs_folder in COMPANIES.items():
            symbol = f"{ticker.upper()}.US"
            company_name, _ = SYMBOLS[symbol]

            layer_news = fetch_layer_news(ticker, date)
            eod_news = fetch_eod_news(symbol, date)
            benzinga_news = fetch_benzinga_news(ticker, date)
            finnhub_news = fetch_finnhub_news(ticker.upper(), date)
            alpha_vantage_news = fetch_alpha_vantage_news(ticker.upper(), date)
            stock_news = fetch_stock_news(ticker, 3,1)

            combined_news = (
                layer_news + eod_news + benzinga_news + finnhub_news + alpha_vantage_news + stock_news
            )

            save_to_gcs_grouped_by_date(combined_news, company_name, gcs_folder)

        current_date += datetime.timedelta(days=1)

# Run the script for a given date range
if __name__ == "__main__":
    start_date = datetime.datetime(2024, 11, 18)
    end_date = datetime.datetime(2024, 12, 2)
    process_news(start_date, end_date)
