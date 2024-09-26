import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load datasets
print("Loading datasets...", flush=True)
df_fake = pd.read_csv(r"E:\fake_news_detection\Datasets\Fake.csv")
df_true = pd.read_csv(r"E:\fake_news_detection\Datasets\True.csv")
print("Datasets loaded successfully.", flush=True)

# Insert target column
df_fake["class"] = 0
df_true["class"] = 1

# Remove last 10 rows for manual testing
df_fake = df_fake.iloc[:-10]
df_true = df_true.iloc[:-10]

# Merge datasets and shuffle
print("Merging and shuffling datasets...", flush=True)
df = pd.concat([df_fake, df_true]).sample(frac=1).reset_index(drop=True)
df = df.drop(["title", "subject", "date"], axis=1)
print("Datasets merged and shuffled.", flush=True)

# Function for text cleaning
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Clean the text
print("Cleaning text data...", flush=True)
df["text"] = df["text"].apply(wordopt)
print("Text data cleaned.", flush=True)

# Define features and labels
x = df["text"]
y = df["class"]

# Split data
print("Splitting data into training and testing sets...", flush=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
print("Data split completed.", flush=True)

# Convert text to vectors
print("Converting text to numerical vectors using TF-IDF...", flush=True)
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)
print("Text vectorization completed.", flush=True)

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Vectorizer saved as 'vectorizer.pkl'.", flush=True)

# Logistic Regression Model
print("Training Logistic Regression model...", flush=True)
LR = LogisticRegression()
LR.fit(xv_train, y_train)
joblib.dump(LR, 'logistic_regression_model.pkl')  # Save model
print("Logistic Regression model trained and saved as 'logistic_regression_model.pkl'.", flush=True)

# Random Forest Classifier Model
print("Training Random Forest Classifier model...", flush=True)
RFC = RandomForestClassifier(random_state=42)
RFC.fit(xv_train, y_train)
joblib.dump(RFC, 'random_forest_model.pkl')  # Save model
print("Random Forest Classifier model trained and saved as 'random_forest_model.pkl'.", flush=True)

print("All models trained and saved successfully!", flush=True)
