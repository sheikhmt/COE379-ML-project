import json
import matplotlib.pyplot as plt

# Define file paths and labels
companies = ["Raytheon", "Honeywell", "Lockheed Martin"]
date_ranges = ["6_Months", "1_Year", "2_Years", "3_Years"]
file_paths = [
    f"{company}_{date_range}_articles.json"
    for company in companies
    for date_range in date_ranges
]

# Set the token limit approximation (for FinBERT)
word_limit = 1500  # Approximate max word count per article for FinBERT compatibility

# Loop through each file path, calculate word counts, count articles compatible with FinBERT, and plot histograms
for file_path in file_paths:
    try:
        # Load the JSON data from file
        with open(file_path, "r") as file:
            articles_data = json.load(file)

        # Calculate word counts for each article's content
        word_counts = [
            len(article["content"].split())
            for article in articles_data
            if article["content"]
        ]

        # Count articles within FinBERT's token limit
        finbert_compatible_count = sum(
            1 for count in word_counts if count <= word_limit
        )

        # Print results for current file
        print(f"{file_path}:")
        print(f"Total number of articles: {len(word_counts)}")
        print(f"Articles within FinBERT limit: {finbert_compatible_count}\n")

        # Plot the histogram for word counts
        plt.figure(figsize=(10, 6))
        plt.hist(word_counts, bins=20, color="skyblue", edgecolor="black")
        plt.title(f"Word Count Distribution for {file_path.replace('_', ' ')}")
        plt.xlabel("Word Count")
        plt.ylabel("Number of Articles")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    except FileNotFoundError:
        print(f"File not found: {file_path}")
