import requests
import json
from datetime import datetime, timedelta
from secret import benzinga_key  # Import Benzinga API key from secret.py

# API settings
benzinga_url = "https://api.benzinga.com/api/v2/news"

# Define companies and their corresponding file paths
COMPANIES = {
    "rtx": "raytheon",
    "noc": "northrop",
    "lmt": "lockheed",
}

# Helper function to get business days
def get_business_days(start_date, end_date):
    current_date = start_date
    business_days = []
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday to Friday
            business_days.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    if not business_days:
        print("It is not a business day. No news was fetched.")
    return business_days

# Fetch news from Benzinga API using the provided request format
def fetch_benzinga_news(ticker, date):
    headers = {
        "accept": "application/json",
    }
    params = {
        "token": benzinga_key,
        "tickers": ticker.upper(),
        "date": date,  # Format: YYYY-MM-DD
        "displayOutput": "full",
        "pageSize": 100,
    }
    response = requests.request("GET", benzinga_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()  # Return the JSON response
    else:
        print(f"Benzinga API error for {ticker}: {response.status_code} - {response.text}")
        return []

# Filter and truncate summaries
def filter_and_truncate_summary(summary):
    """
    Exclude articles with less than 100 characters or truncate those with more than 1500 characters.
    :param summary: The article's summary/body text
    :return: Processed summary or None if it doesn't meet the criteria
    """
    if not summary:
        return None
    if len(summary) < 100:
        return None
    if len(summary) > 1500:
        return summary[:1500] + "..."
    return summary

# Format the date to `YYYY_MM_DD`
def format_date_to_underscore(date_str):
    """
    Convert date string from 'Fri, 29 Nov 2024 16:00:33 -0400' to 'YYYY_MM_DD'.
    :param date_str: Original date string
    :return: Formatted date string in 'YYYY_MM_DD'
    """
    try:
        return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z").strftime("%Y_%m_%d")
    except ValueError:
        print(f"Error parsing date: {date_str}")
        return ""

# Save news data to JSON file
def save_to_json(data, company_name, date):
    if data:
        # Format file name as company_name-YYYY_MM_DD.json
        file_name = f"{company_name}-{datetime.strptime(date, '%Y-%m-%d').strftime('%Y_%m_%d')}.json"
        with open(file_name, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Saved news data for {company_name} on {date} to {file_name}")
    else:
        print(f"No news data found for {company_name} on {date}.")

# Main function to process and save Benzinga news
def process_benzinga_news(start_date, end_date):
    # Get business days between start_date and end_date
    business_days = get_business_days(start_date, end_date)

    # Loop through each date and fetch news for all companies
    for date in business_days:
        print(f"Fetching news for {date}...")
        for ticker, company_name in COMPANIES.items():
            # Fetch Benzinga API news
            benzinga_data = fetch_benzinga_news(ticker, date)
            filtered_news = []
            for article in benzinga_data:
                # Parse and format the "created" date field to `YYYY_MM_DD`
                created_date = format_date_to_underscore(article.get("created", ""))
                summary = filter_and_truncate_summary(article.get("body", ""))
                if summary:  # Include the article only if the summary meets the criteria
                    filtered_news.append(
                        {
                            "Date": created_date,
                            "Title": article.get("title"),
                            "Summary": summary,
                            "URL": article.get("url"),
                            "Source": "Benzinga API",
                        }
                    )

            # Save filtered news to a JSON file
            save_to_json(filtered_news, company_name, date)

# Run the script for a given date range
if __name__ == "__main__":
    start_date = datetime(2024, 12, 3)  # Adjust the start date
    end_date = datetime(2024, 12, 3)    # Adjust the end date
    process_benzinga_news(start_date, end_date)
