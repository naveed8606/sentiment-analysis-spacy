import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load model
nlp = spacy.load("en_core_web_sm")

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/naveed8606/sentiment-analysis-spacy/main/data/sample.csv"
df = pd.read_csv(url)

# Preprocess
def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

df["cleaned"] = df["text"].apply(preprocess_text)

# Vectorize
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df["cleaned"])
y = df["label"]

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Test prediction
def predict_sentiment(text):
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    return model.predict(vectorized)[0]

print(predict_sentiment("This movie was amazing"))
print(predict_sentiment("Worst movie ever"))
