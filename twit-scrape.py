# import csv
# import time
# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# # Set up Chrome options
# chrome_options = Options()
# chrome_options.add_argument("--headless")  # Run headless Chrome (without a GUI)

# driver_path = "D:/Coding/chromedriver-win64/chromedriver-win64/chromedriver.exe"
# service = Service(driver_path)

# # Initialize the Chrome WebDriver
# driver = webdriver.Chrome(service=service, options=chrome_options)

# # List of companies to search
# companies = ["Raytheon", "Lockheed Martin", "Honeywell"]
# # Create a list to hold tweets
# tweets_data = []

# for company in companies:
#     # Open Twitter explore page
#     search_url = f"https://twitter.com/explore"
#     driver.get(search_url)
#     time.sleep(3)  # Give the page some time to load

#     try:
#         # Wait until the search box is present
#         search_box = WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located(
#                 (By.XPATH, '//input[@aria-label="Search query"]')
#             )
#         )
#         # Enter the search query
#         search_box.clear()  # Clear any existing text
#         search_box.send_keys(company)
#         search_box.send_keys(Keys.RETURN)

#         # Wait for the search results to load
#         time.sleep(3)

#         # Scroll to load more tweets
#         body = driver.find_element("tag name", "body")
#         for _ in range(5):  # Scroll multiple times to load more tweets
#             body.send_keys(Keys.PAGE_DOWN)
#             time.sleep(2)

#         # Find tweet elements
#         tweets = driver.find_elements("css selector", "article")

#         # Extract tweet data
#         for tweet in tweets:
#             try:
#                 text = tweet.find_element("css selector", "div[lang]").text
#                 timestamp = tweet.find_element("css selector", "time").get_attribute(
#                     "datetime"
#                 )
#                 tweets_data.append({"timestamp": timestamp, "text": text})
#             except Exception as e:
#                 print("Error extracting tweet:", e)
#     except Exception as e:
#         print(f"Error finding search box for {company}: {e}")

# # Save the tweets to a CSV file
# with open("tweets_data.csv", mode="w", newline="", encoding="utf-8") as file:
#     writer = csv.DictWriter(file, fieldnames=["timestamp", "text"])
#     writer.writeheader()
#     writer.writerows(tweets_data)

# print(f"Scraped {len(tweets_data)} tweets and saved to tweets_data.csv")

# # Close the driver
# driver.quit()
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
from secret import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET


class TwitterClient(object):
    """
    Generic Twitter Class for sentiment analysis.
    """

    def __init__(self):
        """
        Class constructor or initialization method.
        """
        # keys and tokens from the Twitter Dev Console
        consumer_key = CONSUMER_KEY
        consumer_secret = CONSUMER_SECRET
        access_token = ACCESS_TOKEN
        access_token_secret = ACCESS_TOKEN_SECRET

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        """
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        """
        return " ".join(
            re.sub(
                "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet
            ).split()
        )

    def get_tweet_sentiment(self, tweet):
        """
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        """
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return "positive"
        elif analysis.sentiment.polarity == 0:
            return "neutral"
        else:
            return "negative"

    def get_tweets(self, query, count=10):
        """
        Main function to fetch tweets and parse them.
        """
        # empty list to store parsed tweets
        tweets = []

        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search_tweets(q=query, count=count)

            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}

                # saving text of tweet
                parsed_tweet["text"] = tweet.text
                # saving sentiment of tweet
                parsed_tweet["sentiment"] = self.get_tweet_sentiment(tweet.text)

                # appending parsed tweet to tweets list
                if tweet.retweet_count == 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            # return parsed tweets
            return tweets

        except tweepy.TweepyException as e:
            # print error (if any)
            print("Error : " + str(e))


def main():
    # creating object of TwitterClient Class
    api = TwitterClient()
    # calling function to get tweets
    tweets = api.get_tweets(query="Raytheon", count=200)

    # picking positive tweets from tweets
    ptweets = [tweet for tweet in tweets if tweet["sentiment"] == "positive"]
    # percentage of positive tweets
    print("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet["sentiment"] == "negative"]
    # percentage of negative tweets
    print("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))
    # percentage of neutral tweets
    print(
        "Neutral tweets percentage: {} % \
        ".format(
            100 * (len(tweets) - (len(ntweets) + len(ptweets))) / len(tweets)
        )
    )

    # printing first 5 positive tweets
    print("\n\nPositive tweets:")
    for tweet in ptweets[:10]:
        print(tweet["text"])

    # printing first 5 negative tweets
    print("\n\nNegative tweets:")
    for tweet in ntweets[:10]:
        print(tweet["text"])


if __name__ == "__main__":
    # calling main function
    main()
