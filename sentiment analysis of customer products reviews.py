# Import necessary libraries
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Example product reviews
reviews = [
    "This product is amazing! It exceeded all my expectations.",
    "The quality is terrible, I regret buying it.",
    "Decent product, works as expected.",
    "I'm very satisfied with this purchase, highly recommended!",
    "Worst product ever, waste of money."
]

# Create a DataFrame to store reviews and their sentiment scores
df = pd.DataFrame(reviews, columns=['Review'])

# Function to compute sentiment scores
def get_sentiment_score(text):
    score = sia.polarity_scores(text)
    return score['compound']  # Compound score gives overall sentiment

# Apply the function to get sentiment scores for each review
df['Sentiment Score'] = df['Review'].apply(get_sentiment_score)

# Classify sentiment based on score (positive, negative, neutral)
df['Sentiment'] = df['Sentiment Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

# Print the DataFrame with sentiment results
print(df)
