import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# ── Download required NLTK data ────────────────────────────────────
nltk.download('stopwords', quiet=True)

# ── Load dataset ───────────────────────────────────────────────────
df = pd.read_csv(r'D:\New folder\IMDB Dataset.csv')
print(f"✅ Loaded {len(df)} reviews")

# ── Clean text ─────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)         # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)     # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return words

# ── Convert to NLTK format: ({word: True}, label) ─────────────────
def get_features(words):
    return {word: True for word in words}

print("Preparing data... (takes 1-2 mins)")

data = []
for _, row in df.iterrows():
    words = clean_text(row['review'])
    features = get_features(words)
    label = row['sentiment']        # 'positive' or 'negative'
    data.append((features, label))

# ── Split 80% train, 20% test ──────────────────────────────────────
split = int(len(data) * 0.8)
train_data = data[:split]
test_data  = data[split:]

# ── Train Naive Bayes ──────────────────────────────────────────────
print("Training model...")
classifier = NaiveBayesClassifier.train(train_data)

# ── Evaluate ───────────────────────────────────────────────────────
acc = accuracy(classifier, test_data)
print(f"\n✅ Done!")
print(f"Accuracy : {acc * 100:.2f}%")
print("\nTop 10 most informative words:")
classifier.show_most_informative_features(10)

import pickle

# Save classifier and word list
with open(r'D:\New folder\sentiment_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("✅ Model saved!")