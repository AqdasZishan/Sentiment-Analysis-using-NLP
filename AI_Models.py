import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

# Download necessary NLTK corpora and models
#nltk.download('punkt')
#nltk.download('stopwords')

# Load dataset
df = pd.read_csv('Dataset.csv', encoding='ISO-8859-1')

# Remove rows with NaN values
df.dropna(inplace=True)

# Convert sentiment to binary
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Remove HTML tags
def remove_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

df['review'] = df['review'].apply(remove_tags)

# Remove punctuation and convert to lowercase
def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table).lower()

df['review'] = df['review'].apply(remove_punct)

# Remove stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = word_tokenize(text)
    return ' '.join([w for w in words if w not in stop_words])

df['review'] = df['review'].apply(remove_stopwords)
    
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], random_state=0)

# Vectorize the text using Bag-of-Words model
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

# Convert the raw frequency counts into TF-IDF values
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train the Naive Bayes classifier
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Define a function to predict the sentiment using TextBlob
def predict_sentiment_tb(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return 'positive'
    else:
        return 'negative'

# Define a function to predict the sentiment using Machine Learning model
def predict_sentiment_ml(text):
    text_counts = count_vect.transform([text])
    text_tfidf = tfidf_transformer.transform(text_counts)
    return "positive" if clf.predict(text_tfidf)[0] == 1 else "negative"

# Evaluate the TextBlob model on testing set
y_pred_tb = X_test.apply(predict_sentiment_tb)

# convert y_pred_tb from strings to integers
y_pred_tb_int = [0 if label=='neg' else 1 for label in y_pred_tb]

# Evaluate the Machine Learning model on testing set
X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)

def accuracy_tb():
    # compute accuracy score for Textblob using integers
    accuracy_tb = accuracy_score(y_test, y_pred_tb_int)
    return(f"TextBlob Model Accuracy : {accuracy_tb:.4f}\n")

def accuracy_ml():
    # compute accuracy score for Machine Learning Model
    accuracy = accuracy_score(y_test, y_pred)
    return(f"Machine Learning Model Accuracy : {accuracy:.4f}\n")

def stat_report():
    # Show confusion matrix, precision, recall, and F1-score
    stats = "Confusion Matrix :\n" + str(confusion_matrix(y_test, y_pred)) + "\n"
    stats += "\nClassification Report :\n" + str(classification_report(y_test, y_pred))
    return stats
