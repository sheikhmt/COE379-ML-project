import requests
import pandas as pd
import re
from datetime import datetime
from secret import EODHD_key

# Define your EOD Historical Data API key
api_key = EODHD_key  # Replace with your actual API key

# Define the company symbols (Raytheon, Honeywell, Lockheed Martin)
symbols = {
    "RTX.US": "Raytheon",
    "HON.US": "Honeywell",
    "LMT.US": "Lockheed Martin",
}

# Get today's date in the format Y-m-d
today = datetime.utcnow().strftime("%Y-%m-%d")
test_data = "2024-10-15"  # Example for test data

# Construct the URL for Financial News Feed API
url = f"https://eodhd.com/api/sentiments"

# Define parameters for the API call
params = {
    "s": ",".join(symbols.keys()),  # Convert the list of symbols to a comma-separated string
    "api_token": api_key,
    "from": test_data,  # Using test_data for this example
    "to": today,
    "limit": 100,  # Limit to 100 articles (you can change this limit)
    "fmt":"json"
}
data = requests.get(url,params=params).json()

print(data)