import finnhub
import json
import datetime
from google.cloud import storage
from google.oauth2 import service_account
from secret import finnhub_key  # Import your Finnhub API key

# Google Cloud credentials
credentials_path = r"C:\Users\dhn6u\Downloads\coe379-ml-project-8266ccc1ee98.json"
credentials = service_account.Credentials.from_service_account_file(credentials_path)
storage_client = storage.Client(credentials=credentials)

# Finnhub API client
finnhub_client = finnhub.Client(api_key=finnhub_key)

# GCS bucket name
bucket_name = "fin_analysis_data"

# Company tickers and GCS folder paths
COMPANIES = {
    "RTX": "raytheon/news_articles",
    "NOC": "northrop/news_articles",
    "LMT": "lockheed/news_articles",
}

# Function to fetch news data from Finnhub API
def fetch_finnhub_news(ticker, start_date, end_date):
    """
    Fetch company news from Finnhub API.
    :param ticker: Company ticker (e.g., 'AAPL')
    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    :return: List of news articles
    """
    try:
        news = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
        return news
    except Exception as e:
        print(f"Finnhub API error for {ticker}: {e}")
        return []

# Convert Unix timestamp to YYYY-MM-DD format
def convert_unix_to_date(unix_timestamp):
    """
    Convert a 10-digit Unix timestamp to YYYY-MM-DD format.
    :param unix_timestamp: 10-digit Unix timestamp
    :return: Date string in YYYY-MM-DD format
    """
    return datetime.datetime.utcfromtimestamp(unix_timestamp).strftime("%Y-%m-%d")

# Save news data to GCS
def save_to_gcs(data, company_name, gcs_folder, date):
    """
    Save the filtered news data to Google Cloud Storage as JSON.
    :param data: Filtered news data
    :param company_name: Company name for file naming
    :param gcs_folder: GCS folder path
    :param date: Date for the file name
    """
    if data:
        file_name = f"{company_name.lower()}-{date}.json"
        json_data = json.dumps(data, indent=4)
        gcs_path = f"{gcs_folder}/{file_name}"

        # Upload to GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(json_data, content_type="application/json")
        print(f"Uploaded {file_name} to gs://{bucket_name}/{gcs_path}")
    else:
        print(f"No data to upload for {company_name} on {date}")

# Main function to fetch and process news
def process_finnhub_news(start_date, end_date):
    """
    Process news data for all companies in the given date range.
    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    """
    # Loop through each company and fetch news
    for ticker, gcs_folder in COMPANIES.items():
        print(f"Fetching Finnhub news for {ticker}...")
        news_data = fetch_finnhub_news(ticker, start_date, end_date)

        # Filter relevant fields
        filtered_news = [
            {
                "Date": convert_unix_to_date(article.get("datetime", 0)),
                "Title": article.get("headline", ""),
                "Summary": article.get("summary", ""),
                "URL": article.get("url", ""),
                "Source": "Finnhub API",
            }
            for article in news_data
        ]

        # Save to GCS
        save_to_gcs(filtered_news, ticker, gcs_folder, start_date)

# Run the script for a given date range
if __name__ == "__main__":
    # Define the date range
    start_date = "2024-11-29"
    end_date = "2024-12-02"

    # Process news data for all companies
    process_finnhub_news(start_date, end_date)
