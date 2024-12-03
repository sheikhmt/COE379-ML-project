from time import sleep
import requests
import os
import json
import datetime
from google.cloud import storage
from google.oauth2 import service_account
from secret import layer_key  # Ensure this module exists and contains your API key

# Google Cloud credentials
credentials_path = r"C:\Users\dhn6u\Coding\COE379-ML-project\coe379-ml-project-8266ccc1ee98.json"
credentials = service_account.Credentials.from_service_account_file(credentials_path)
storage_client = storage.Client(credentials=credentials)

# Define LayerAPI credentials and endpoints
API_KEY = layer_key  # Replace with your actual LayerAPI key
BASE_URL = "https://api.apilayer.com/financelayer/news"  # LayerAPI endpoint

# Define bucket and directories for each company
BUCKET_NAME = "fin_analysis_data"
COMPANIES = {
    "rtx": "raytheon/news_articles",
    "noc": "northrop/news_articles",
    "lmt": "lockheed/news_articles",
}

def fetch_layerapi_news(company, start_date, end_date):
    """Fetch news articles for a specific company from LayerAPI."""
    headers = {
        "apikey": API_KEY,
    }

    params = {
        "tickers": company,  # Pass the company's ticker (e.g., 'rtx')
        "date_from": start_date,
        "date_to": end_date,
        "limit": 100,  # Limit the number of articles fetched
    }

    response = requests.get(BASE_URL, headers=headers, params=params)

    if response.status_code == 200:
        news_data = response.json().get("data", [])
        filtered_news = [
            {
                "headline": item.get("title", ""),
                "summary": item.get("description", ""),
                "url": item.get("url", ""),
            }
            for item in news_data
        ]
        return filtered_news
    else:
        print(
            f"Error fetching news for {company}: {response.status_code} - {response.text}"
        )
        return []


def save_to_gcs(company, data):
    """Save the news data as a JSON file to Google Cloud Storage."""
    # Create the JSON file name based on current date
    file_name = f"{company}_{datetime.datetime.now().strftime('%Y-%m-%d')}_news.json"

    # Convert data to JSON string
    json_data = json.dumps(data, indent=4)

    # Define the blob (file) path
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"fin_text/{COMPANIES[company]}/{file_name}")

    # Upload JSON data to the bucket
    blob.upload_from_string(json_data, content_type="application/json")
    print(f"Uploaded {file_name} to gs://{BUCKET_NAME}/{COMPANIES[company]}/")


def main():
    # Define the date range (last 24 hours, for example)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=1)

    # Format the dates as strings (YYYY-MM-DD)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Loop over each company to fetch and upload news
    for company in COMPANIES.keys():
        try:
            print(
                f"Fetching news for {company.upper()} from {start_date_str} to {end_date_str}..."
            )
            news_data = fetch_layerapi_news(company, start_date_str, end_date_str)

            if news_data:
                print(f"Saving {len(news_data)} articles for {company.upper()} to GCS...")
                save_to_gcs(company, news_data)
            else:
                print(
                    f"No news articles found for {company.upper()} during this period."
                )

        except Exception as e:
            print(f"Error processing {company.upper()}: {e}")

        # Sleep to avoid hitting API rate limits (if necessary)
        sleep(5)


if __name__ == "__main__":
    main()
