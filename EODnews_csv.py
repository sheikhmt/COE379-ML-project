import requests
import json
from datetime import datetime
from google.cloud import storage
from google.oauth2 import service_account
from secret import EODHD_key  # Ensure this module contains your API key

# Define the API key
api_key = EODHD_key

# Google Cloud credentials
credentials_path = r"C:\Users\dhn6u\Downloads\coe379-ml-project-8266ccc1ee98.json"
credentials = service_account.Credentials.from_service_account_file(credentials_path)
storage_client = storage.Client(credentials=credentials)
bucket_name = "fin_analysis_data"

# Define company symbols and GCS directories
symbols = {
    "RTX.US": ("Raytheon", "raytheon/news_articles"),
    "NOC.US": ("Northrop", "northrop/news_articles"),
    "LMT.US": ("Lockheed Martin", "lockheed/news_articles"),
}

# Define the EODHD API URL
url = "https://eodhistoricaldata.com/api/news"

# Date range for the API request
start_date = "2024-12-01"  # Example start date
end_date = datetime.utcnow().strftime("%Y-%m-%d")  # Today's date

# Function to fetch news data for a specific symbol
def fetch_news_data(symbol):
    params = {
        "s": symbol,
        "api_token": api_key,
        "from": start_date,
        "to": end_date,
        "limit": 100,  # Adjust limit as needed
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()  # Return JSON data
    else:
        print(f"Error fetching data for {symbol}: {response.status_code} - {response.text}")
        return []

# Function to upload JSON data to Google Cloud Storage
def upload_to_gcs(data, company_name, gcs_folder, date):
    if data:
        # Convert data to JSON string
        json_data = json.dumps(data, indent=4)

        # Define the GCS path and file name
        file_name = f"{company_name}_news_{date}.json"
        gcs_path = f"{gcs_folder}/{file_name}"

        # Upload JSON data to GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(json_data, content_type="application/json")
        print(f"Uploaded {file_name} to gs://{bucket_name}/{gcs_path}")
    else:
        print(f"No data to upload for {company_name}")

# Function to organize news data by date and upload them to GCS
def save_news_by_date(data, company_name, gcs_folder):
    if data:
        # Organize articles by date (first 10 characters of the date string)
        articles_by_date = {}
        for article in data:
            article_date = article.get("date", "")[:10]  # Extract the YYYY-MM-DD part of the date
            if article_date:
                if article_date not in articles_by_date:
                    articles_by_date[article_date] = []
                articles_by_date[article_date].append({
                    "Date": article.get("date"),
                    "Title": article.get("title", ""),
                    "Content": article.get("content", ""),
                    "Link": article.get("link", ""),
                    "Source": "EOD API",
                })

        # Upload each date's articles to GCS
        for article_date, articles in articles_by_date.items():
            upload_to_gcs(articles, company_name, gcs_folder, article_date)
    else:
        print(f"No data to process for {company_name}")

# Main script to fetch, organize, and upload news data for each symbol
def main():
    for symbol, (company_name, gcs_folder) in symbols.items():
        # Fetch news data for the current symbol
        print(f"Fetching news for {symbol} ({company_name})...")
        news_data = fetch_news_data(symbol)

        # Organize and upload news data by date
        save_news_by_date(news_data, company_name, gcs_folder)

if __name__ == "__main__":
    main()
