from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time

# Keywords for searching tweets
query = "(Raytheon OR Honeywell OR LockheedMartin) (defense OR contract OR military OR earnings)"

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run headless Chrome (without a GUI)

driver_path = "C:/Users/arish/Coding/chromedriver-win64/chromedriver.exe"
service = Service(driver_path)

# Initialize the Chrome WebDriver with the service object
driver = webdriver.Chrome(service=service, options=chrome_options)

# Open Twitter search page
search_url = f"https://twitter.com/search?q={query}&f=live"
driver.get(search_url)

# Give the page some time to load
time.sleep(5)

# Scroll down to load more tweets
body = driver.find_element("tag name", "body")
for _ in range(5):  # Scroll multiple times to load more tweets
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(3)  # Increased sleep time to ensure tweets load

# Find tweet elements
tweets = driver.find_elements("css selector", "article")

# Check if tweets were found
print(f"Found {len(tweets)} tweets")

# Extract and print the tweet data if any tweets are found
if len(tweets) > 0:
    for tweet in tweets:
        try:
            # Extract the tweet text
            text = tweet.find_element("css selector", "div[lang]").text
            # Extract the timestamp (the 'time' tag holds the tweet time)
            timestamp = tweet.find_element("css selector", "time").get_attribute(
                "datetime"
            )
            print(f"{timestamp}: {text}\n")
        except Exception as e:
            # Handle case where an element is missing
            print("Error extracting tweet:", e)
else:
    print("No tweets found. Check if the CSS selectors need updating.")

# Close the driver
driver.quit()
