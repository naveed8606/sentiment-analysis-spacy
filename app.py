import streamlit as st
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load dataset
url = "https://raw.githubusercontent.com/naveed8606/sentiment-analysis-spacy/main/data/sample.csv"
df = pd.read_csv(url)

# Preprocess function
def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# Prepare data
df["cleaned"] = df["text"].apply(preprocess_text)

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Streamlit UI
st.title("🎬 Sentiment Analysis")
st.write("Enter a sentence and check its sentiment")

user_input = st.text_area("Enter your response here:")

if st.button("Analyze"):
    if user_input.strip() != "":
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == "positive":
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")
    else:
        st.warning("Please enter some text")