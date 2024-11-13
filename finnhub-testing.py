import finnhub
from datetime import datetime, timedelta
from secret import FINHUB_KEY

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINHUB_KEY)


# Function to get article count within a date range
def get_article_count(company, start_date, end_date):
    news_data = finnhub_client.company_news(company, _from=start_date, to=end_date)
    return len(news_data)


# Main function to test different date ranges
def test_article_counts(company="RTX"):
    end_date = datetime.today()
    periods = {
        "1 Months": end_date - timedelta(days=1 * 30),
        "6 Months": end_date - timedelta(days=6 * 30),
        "9 Months": end_date - timedelta(days=9 * 30),
        "1 Years": end_date - timedelta(days=1 * 364),
    }

    print(f"Article counts for {company}:")
    for period, start_date in periods.items():
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        print(f"{period} ({start_date_str} - {end_date_str}):")
        article_count = get_article_count(company, start_date_str, end_date_str)
        print(f"{period} - {article_count} articles")


if __name__ == "__main__":
    test_article_counts()
