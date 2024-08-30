import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Define paths
data_dir = 'data'
resume_csv_path = os.path.join(data_dir, 'Resume.csv')
train_data_path = os.path.join(data_dir, 'train_data.csv')
test_data_path = os.path.join(data_dir, 'test_data.csv')
vectorizer_path = os.path.join(data_dir, 'vectorizer.joblib')

# Load the dataset
resume_data = pd.read_csv(resume_csv_path)

# Preprocess the data
resume_data['cleaned_resume'] = resume_data['Resume_str'].apply(lambda x: " ".join(x.lower().split()))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(resume_data['cleaned_resume'], resume_data['Category'], test_size=0.2, random_state=42)

# Convert the text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, vectorizer_path)

# Save the processed data
X_train_df = pd.DataFrame(X_train_tfidf.toarray())
X_train_df['Category'] = y_train.values
X_train_df.to_csv(train_data_path, index=False)

X_test_df = pd.DataFrame(X_test_tfidf.toarray())
X_test_df['Category'] = y_test.values
X_test_df.to_csv(test_data_path, index=False)

print("Data preprocessing complete. Processed data and vectorizer saved.")
