# COE379 ML Project

**Machine Learning Pipeline for Stock Market Prediction Using Sentiment Analysis**

---

## 📌 **Project Overview**

This repository hosts an end-to-end Machine Learning pipeline developed to predict stock market performance by leveraging sentiment analysis from news, Reddit discussions, and Twitter posts. It integrates NLP techniques with predictive analytics to explore relationships between public sentiment and stock movements.

---

## 🚀 **Features**

- **Data Collection** from multiple online sources:
  - Finnhub (financial news)
  - Reddit (Pushshift API)
  - Twitter (`snscrape`)

- **Sentiment Analysis** using NLP tools:
  - VADER (Valence Aware Dictionary and sEntiment Reasoner)
  - FinBERT (Financial-domain transformer-based model)

- **Market Data Integration** for sentiment correlation with stock price movements.

- **Machine Learning Models** to predict future stock trends.

---

## 📁 **Project Structure**

```plaintext
.
├── data/                            
│   ├── Honeywell_news_highlights.csv
│   ├── Lockheed Martin_news_highlights.csv
│   ├── Raytheon_news_highlights.csv
│   └── news_highlights.csv
│
├── scripts/                         
│   ├── EODpull_draft.py
│   ├── EODpull_sentiment.py
│   ├── collectstockprice.py
│   ├── marketpull.py
│   ├── news-finnhub.py
│   ├── news-scrape.py
│   ├── twit-scrape.py
│   └── extract_reddit.ipynb
│
├── analysis/
│   └── old-twitter-analysis.py
│
├── requirements.txt
└── README.md
```
# ⚙️ Installation

## Clone the Repository
```bash
git clone https://github.com/sheikhmt/COE379-ML-project.git
cd COE379-ML-project
```
## Fetch News Data
```
python scripts/news-finnhub.py --symbol HON --from 2025-05-01 --to 2025-05-31
```
## Fetch Twitter Data
```
python scripts/twit-scrape.py --query "Honeywell" --days 30
```
## Combine Sentiment & Market Data

```
python scripts/EODpull_sentiment.py --symbol HON --date 2025-05-30
```
# 📦 Dependencies

- Python (≥3.9)
- pandas
- requests
- BeautifulSoup4
- snscrape
- vaderSentiment
- transformers (FinBERT)
