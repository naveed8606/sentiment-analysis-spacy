# Sentiment Analysis using spaCy

## 📌 Overview

This project performs sentiment classification (positive/negative) using Natural Language Processing (NLP) techniques with spaCy and machine learning.

* Used IMDb dataset (5000 samples)
* Preprocessing with spaCy (lemmatization, stopword removal)
* TF-IDF vectorization
* Logistic Regression model
* Achieved ~84% accuracy

---

## 🛠 Tech Stack

* Python
* spaCy
* scikit-learn
* pandas

---

## ⚙️ Features

* Text preprocessing (lemmatization, stopword removal)
* TF-IDF feature extraction
* Logistic Regression classifier
* Model evaluation (accuracy, classification report)

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/naveed8606/sentiment-analysis-spacy.git
cd sentiment-analysis-spacy
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 4. Run the project

```bash
python main.py
```

---

##  Sample Output

```
Accuracy: ~0.84

This movie was amazing → positive  
Worst movie ever → negative
```

---

## 🚀 Future Improvements

* Use deep learning models (LSTM / Transformers)
* Deploy using Streamlit
* Improve dataset diversity
