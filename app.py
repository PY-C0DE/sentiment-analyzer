import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

# ── Load saved model ───────────────────────────────────────────────
with open(r'D:\New folder\sentiment_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

# ── Same clean function as before ──────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return {word: True for word in words}

# ── Streamlit UI ───────────────────────────────────────────────────
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Type any movie review below and the AI will predict if it's positive or negative.")

user_input = st.text_area("Enter your review here:", height=150,
                           placeholder="e.g. This movie was absolutely brilliant...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        features = clean_text(user_input)
        prediction = classifier.classify(features)
        prob = classifier.prob_classify(features)

        pos_score = round(prob.prob('positive') * 100, 2)
        neg_score = round(prob.prob('negative') * 100, 2)

        if prediction == 'positive':
            st.success(f"✅ Positive Review  —  {pos_score}% confidence")
        else:
            st.error(f"❌ Negative Review  —  {neg_score}% confidence")

        st.divider()
        st.write("**Confidence Scores:**")
        st.progress(pos_score / 100, text=f"Positive: {pos_score}%")
        st.progress(neg_score / 100, text=f"Negative: {neg_score}%")