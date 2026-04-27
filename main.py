import pandas as pd
import spacy
from src.preprocess import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

# Load data
df = pd.read_csv("data/sample.csv")

df["cleaned"] = df["text"].apply(preprocess_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]
