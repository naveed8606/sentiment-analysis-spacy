import streamlit as st
import spacy
import random
import csv
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬", layout="centered")

st.sidebar.title("ℹ️ About")
st.sidebar.write("""
This app analyzes movie reviews and predicts sentiment.

Model: Logistic Regression  
Data: IMDb Dataset  
""")

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

@st.cache_resource
def load_model():
    nlp = spacy.load("en_core_web_sm")

    url = "https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/IMDB-Dataset.csv"
    
    response = requests.get(url)
    lines = response.text.splitlines()
    reader = csv.DictReader(lines)

    data = list(reader)

    texts = [row["review"] for row in data][:20000]
    labels = [row["sentiment"] for row in data][:20000]

    def preprocess_text(text):
        doc = nlp(text)
        return " ".join([
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct
        ])

    cleaned = [preprocess_text(t) for t in texts]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        max_df=0.9,
        min_df=5
    )

    X = vectorizer.fit_transform(cleaned)
    y = labels

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    return nlp, vectorizer, model

nlp, vectorizer, model = load_model()

def preprocess_text(text):
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ])

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
        "An outstanding performance by the entire cast."
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
        "One of the worst films I have ever seen."
    ]
}

st.markdown("<h1 style='text-align: center;'>🎬 Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Analyze movie reviews instantly using NLP</p>", unsafe_allow_html=True)

st.divider()

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

with st.form(key="sentiment_form"):
    user_input = st.text_area(
        "Enter your review:",
        value=st.session_state.user_input,
        height=150,
        placeholder="Type your review here..."
    )

    submit_button = st.form_submit_button("🔍 Analyze")

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