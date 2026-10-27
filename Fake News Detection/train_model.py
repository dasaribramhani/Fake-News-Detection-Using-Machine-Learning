import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# 1. File Path Set-up
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "dataset", "news.csv")

# 2. Load dataset
data = pd.read_csv(csv_path, on_bad_lines='skip')
print("COLUMN NAMES:",data.columns.tolist())
print("LABEL VALUES:",data['label'].value_counts())
print(data.groupby('label')['text'].first())

# 3. Clean empty rows & clean labels (0 and 1 only)
data = data.dropna(subset=['text', 'label'])
data['label'] = data['label'].astype(str).str.extract(r'(\d+)')[0]
data = data.dropna(subset=['label'])
data['label'] = data['label'].astype(int)
data = data[data['label'].isin([0, 1])]

# Features and labels
X = data["text"]
y = data["label"]

# 4. Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english', max_df=0.7)),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 5. Train model
model.fit(X, y)

# 6. Save model in 'model' directory
os.makedirs(os.path.join(base_dir, "model"), exist_ok=True)
model_path = os.path.join(base_dir, "model", "fake_news_model.pkl")

joblib.dump(model, model_path)

print("Model trained successfully!")
print(f"Model saved in {model_path}")