# Import necessary libraries
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load pre-trained BERT model and tokenizer from Hugging Face
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Initialize sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Example customer reviews
reviews = [
    "This product is amazing! It exceeded all my expectations.",
    "The quality is terrible, I regret buying it.",
    "Decent product, works as expected.",
    "I'm very satisfied with this purchase, highly recommended!",
    "Worst product ever, waste of money."
]

# Convert reviews into a DataFrame
df = pd.DataFrame(reviews, columns=['Review'])

# Analyze the sentiment of each review using BERT
df['Sentiment'] = df['Review'].apply(lambda review: sentiment_analyzer(review)[0]['label'])

# Display the DataFrame with sentiments
print(df)
