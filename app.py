import streamlit as st
import pandas as pd
import spacy
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

#PAGE CONFIG 
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬", layout="centered")

# SIDEBAR
st.sidebar.title("ℹ️ About")
st.sidebar.write("""
This app analyzes movie reviews and predicts sentiment.

Model: Logistic Regression  
Data: IMDb Dataset  
""")

#SESSION STATE
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# LOAD MODEL (CACHED)
@st.cache_resource
def load_model():
    nlp = spacy.load("en_core_web_sm")

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

nlp, vectorizer, model = load_model()

#PREPROCESS
def preprocess_text(text):
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ])

#EXAMPLES
examples = {
    "Positive": [
        "Amazing movie!",
        "Loved it!",
        "Great acting and story.",
        "Fantastic experience.",
        "Very enjoyable film.",
        "This movie was absolutely brilliant from start to finish.",
        "The acting, direction, and storyline were all top-notch.",
        "A masterpiece with emotional depth and stunning visuals.",
        "I was completely engaged and loved every moment of it.",
        "An outstanding performance by the entire cast.",
        "The film delivers a powerful and inspiring message.",
        "Beautifully executed with a compelling narrative.",
        "One of the best movies I have watched this year.",
        "A perfect blend of drama, emotion, and entertainment.",
        "Highly recommended for anyone who loves good cinema."
    ],
    "Negative": [
        "Terrible movie.",
        "Waste of time.",
        "Very boring.",
        "Bad acting.",
        "Not worth watching.",
        "This movie was extremely disappointing and poorly executed.",
        "The story was weak and the pacing was painfully slow.",
        "I regret watching this, it was a complete waste of time.",
        "The acting was terrible and the plot made no sense.",
        "One of the worst films I have ever seen.",
        "The movie started okay but became unbearable quickly.",
        "Poor direction and a very predictable storyline.",
        "It failed to keep me interested at any point.",
        "Completely overrated and not enjoyable at all.",
        "A dull and lifeless movie experience."
    ]
}

# UI
st.markdown(
    "<h1 style='text-align: center;'>🎬 Sentiment Analysis App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>Analyze movie reviews instantly using NLP</p>",
    unsafe_allow_html=True
)

st.divider()

#EXAMPLE SELECTOR 
st.markdown("### 🔹 Try an example")

category = st.selectbox("Choose type:", ["Positive", "Negative"])

selected_example = st.selectbox(
    "Select a review:",
    examples[category]
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Use Selected Example"):
        st.session_state.user_input = selected_example

with col2:
    if st.button("Random Example"):
        random_category = random.choice(list(examples.keys()))
        st.session_state.user_input = random.choice(examples[random_category])

# INPUT 
with st.form(key="sentiment_form"):
    user_input = st.text_area(
        "Enter your review:",
        value=st.session_state.user_input,
        height=150,
        placeholder="Type your review here..."
    )

    submit_button = st.form_submit_button("🔍 Analyze")

# RESULT
if submit_button:

    if user_input.strip():

        with st.spinner("Analyzing sentiment..."):
            cleaned = preprocess_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            confidence = max(proba)

        st.markdown("## 📊 Result")

        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

        st.metric("Confidence", f"{confidence:.2f}")

    else:
        st.warning("⚠️ Please enter some text")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built using NLP & Machine Learning</p>",
    unsafe_allow_html=True
)