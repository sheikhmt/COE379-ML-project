import tweepy
from secret import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET

# Authentication (fill with your credentials)
consumer_key = CONSUMER_KEY
consumer_secret = CONSUMER_SECRET
access_token = ACCESS_TOKEN
access_token_secret = ACCESS_TOKEN_SECRET

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

# Define search query for defense-related stocks
query = "(#Raytheon OR $RTX OR #Honeywell OR $HON OR #LockheedMartin OR $LMT) (defense OR contract OR earnings)"

# Collect tweets
tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", since="2023-01-01").items(
    1000
)

# Store tweets and sentiment analysis
for tweet in tweets:
    print(tweet.text)
    # You can perform sentiment analysis here
