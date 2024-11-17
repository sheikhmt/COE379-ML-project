import json
import os
import datetime
import finnhub
from google.cloud import storage
from time import sleep
from secret import FINHUB_KEY


finnhub_client = finnhub.Client(api_key=FINHUB_KEY)

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Define bucket and directories for each company
BUCKET_NAME = "fin_analysis_data"
COMPANIES = {
    "RTX": "raytheon/news_articles",
    "HON": "honeywell/news_articles",
    "LMT": "lockheed/news_articles",
}


def get_finnhub_news(company, start_date, end_date):
    """Fetch news articles for a specific company from Finnhub"""
    news_data = finnhub_client.company_news(company, _from=start_date, to=end_date)
    filtered_news = [
        {
            "headline": item.get("headline", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
        }
        for item in news_data
    ]

    return filtered_news


def save_to_gcs(company, data):
    """Save the news data as a JSON file to Google Cloud Storage"""
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
                f"Fetching news for {company} from {start_date_str} to {end_date_str}..."
            )
            news_data = get_finnhub_news(company, start_date_str, end_date_str)

            if news_data:
                print(f"Saving {len(news_data)} articles for {company} to GCS...")
                save_to_gcs(company, news_data)
            else:
                print(f"No news articles found for {company} during this period.")

        except Exception as e:
            print(f"Error processing {company}: {e}")

        # Sleep to avoid hitting API rate limits (if necessary)
        sleep(5)


if __name__ == "__main__":
    main()
