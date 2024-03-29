import pandas as pd

# Replace 'path_to_file' with the actual path to your dataset
path_to_file = 'C:\\Users\\lilbl\\OneDrive\\Desktop\\Sentiment Analysis on Movie Reviews\\IMDB Dataset.csv'
df = pd.read_csv(path_to_file)


print(df.head())

import re

# Define a function to clean the reviews
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = text.strip()
    return text

# Apply the cleaning function to the review column
df['review'] = df['review'].apply(clean_text)

# Check the cleaned reviews
print(df.head())

# Save the cleaned dataset to a new CSV file
df.to_csv('C:\\Users\\lilbl\\OneDrive\\Desktop\\Sentiment Analysis on Movie Reviews\\Cleaned_IMDB_Dataset.csv', index=False)

from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming your DataFrame is named 'df' and the cleaned text column is 'review'

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Fit and transform the 'review' column
X_tfidf = tfidf_vectorizer.fit_transform(df['review'])

# Now 'X_tfidf' is the numerical representation of your text data

from sklearn.model_selection import train_test_split

# Convert the sentiment labels from text to numeric
# 'positive': 1, 'negative': 0
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Save the DataFrame after converting sentiment labels and cleaning the text
df.to_csv('C:\\Users\\lilbl\\OneDrive\\Desktop\\Sentiment Analysis on Movie Reviews\\Cleaned_IMDB_Dataset.csv', index=False)

# Get the target variable
y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize the Logistic Regression model
logistic_model = LogisticRegression()

# Train the model
logistic_model.fit(X_train, y_train)

# Predict on the test set
y_pred_logistic = logistic_model.predict(X_test)

# Calculate and print the accuracy
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f"Accuracy of Logistic Regression: {accuracy_logistic}")

# Print classification report
print(classification_report(y_test, y_pred_logistic))

