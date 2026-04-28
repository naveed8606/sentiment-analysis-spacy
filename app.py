import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Cache model so it doesn't retrain every time
@st.cache_resource
def load_model():
    nlp = spacy.load("en_core_web_sm")

    # Use REAL dataset (IMDb)
    url = "https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/IMDB-Dataset.csv"
    df = pd.read_csv(url)

    df.rename(columns={"review": "text", "sentiment": "label"}, inplace=True)
    df = df.sample(20000, random_state=42)

    def preprocess_text(text):
        doc = nlp(text)
        return " ".join([
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct
        ])

    df["cleaned"] = df["text"].apply(preprocess_text)

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=5000,
        max_df=0.9,
        min_df=5
    )

    X = vectorizer.fit_transform(df["cleaned"])
    y = df["label"]

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    return nlp, vectorizer, model

# Load cached model
nlp, vectorizer, model = load_model()

# Preprocess for user input
def preprocess_text(text):
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ])

# UI
st.title("🎬 Sentiment Analysis App")
st.write("Enter a movie review and check its sentiment")

with st.form(key="sentiment_form"):
    user_input = st.text_area("Enter your review:")
    submit_button = st.form_submit_button("Analyze")

if submit_button:
    if user_input.strip():
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == "positive":
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")
    else:
        st.warning("Please enter some text")