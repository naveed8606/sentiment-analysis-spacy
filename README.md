#  Sentiment Analysis Web App — Powered by spaCy & Machine Learning

![Python](https://img.shields.io/badge/Python-3.11.9-blue?style=for-the-badge&logo=python)
![spaCy](https://img.shields.io/badge/spaCy-3.7.4-09A3D5?style=for-the-badge&logo=spacy)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge&logo=render)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A machine learning web application that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative** — deployed live on the internet.

🔗 **Live App**: [https://sentiment-analysis-spacy-1.onrender.com](https://sentiment-analysis-spacy-1.onrender.com)

---


## ❗ Problem Statement

Businesses receive **thousands of customer reviews every day**. Manually reading and categorizing each one as positive or negative is:

- ⏱️ Time-consuming
- 💸 Expensive
- ❌ Inconsistent and prone to human error

There is a clear need for an **automated, scalable, and intelligent system** that can instantly understand the sentiment behind any piece of text.

---

## ✅ Solution

This app uses **Natural Language Processing (NLP)** and **Machine Learning** to:

1. Accept any movie or product review as input
2. Preprocess the text using spaCy (lemmatization + stopword removal)
3. Convert text into numerical features using TF-IDF vectorization
4. Predict the sentiment as **Positive ✅** or **Negative ❌** in real time

No manual effort. No delays. Instant results.

---

##  Live Demo

🚀 Try the app live here: **[https://sentiment-analysis-spacy-1.onrender.com](https://sentiment-analysis-spacy-1.onrender.com)**

> ⚠️ Note: The app is hosted on Render's free tier. It may take a while to wake up if it has been inactive. Please be patient on the first load.

**Example Inputs to Try:**

| Review | Predicted Sentiment |
|--------|-------------------|
| "This movie was absolutely amazing, I loved every moment!" | ✅ Positive |
| "This was the worst film I've ever seen. Completely boring." | ❌ Negative |
| "The acting was brilliant and the storyline kept me hooked." | ✅ Positive |
| "Terrible script, bad direction. Total waste of time." | ❌ Negative |

---

##  Real World Use Cases

| Industry | Application |
|----------|-------------|
| 🎬 OTT Platforms (Netflix, Prime Video) | Auto-tag and categorize viewer reviews |
| 🛒 E-Commerce (Amazon, Flipkart) | Filter and moderate product feedback |
| 🍽️ Hospitality (Zomato, MakeMyTrip) | Monitor customer satisfaction at scale |
| 🏦 Banking & Fintech | Analyze customer complaints automatically |
| 📱 Social Media | Track brand reputation and public opinion |
| 🏥 Healthcare | Understand patient feedback and improve services |
| 🎓 EdTech | Analyze student reviews for course improvement |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11.9 | Core programming language |
| **NLP** | spaCy 3.7.4 | Text preprocessing — lemmatization & stopword removal |
| **ML Model** | scikit-learn | TF-IDF Vectorization + Logistic Regression |
| **Frontend** | Streamlit | Interactive web UI |
| **Deployment** | Render | Cloud hosting and live deployment |
| **Dataset** | IMDb Movie Reviews | 5000 labeled reviews for training |

---

## 📁 Project Structure

```
sentiment-analysis-spacy/
│
├── app.py                          
├── main.py                         
├── Sentiment_Analysis_spacy.ipynb  
├── requirements.txt               
├── runtime.txt                     
└── README.md                       
```

---

##  How It Works

```
User Input (Raw Review Text)
        │
        ▼
┌─────────────────────────┐
│  spaCy Preprocessing    │
│  - Tokenization         │
│  - Lemmatization        │
│  - Stopword Removal     │
│  - Punctuation Removal  │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  TF-IDF Vectorization   │
│  (scikit-learn)         │
│  Text → Numbers         │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Logistic Regression    │
│  Trained Classifier     │
└─────────────────────────┘
        │
        ▼
   Positive  /   Negative
```

### Step-by-Step Breakdown:

1. **Text Input** — User types a movie review into the Streamlit UI
2. **Preprocessing with spaCy** — The text is tokenized, lemmatized (words reduced to root form), and cleaned by removing stopwords and punctuation
3. **TF-IDF Vectorization** — The cleaned text is transformed into a numerical feature vector
4. **Logistic Regression Prediction** — The trained model predicts the sentiment class
5. **Result Display** — The app displays Positive ✅ or Negative ❌ instantly

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Training Dataset** | IMDb Movie Reviews |
| **Total Samples** | 5,000 reviews |
| **Train/Test Split** | 80% / 20% |
| **Model** | Logistic Regression |
| **Accuracy** | ~84% |

---

## 💻 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/naveed8606/sentiment-analysis-spacy.git
cd sentiment-analysis-spacy
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # On Mac/Linux
venv\Scripts\activate           # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python main.py
```

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

### 6. Open in Browser
```
http://localhost:8501
```

---

## 🚀 Deployment on Render

This app is deployed on **Render** as a Web Service. Here's how it was configured:

| Setting | Value |
|---------|-------|
| **Runtime** | Python |
| **Python Version** | 3.11.9 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| **Environment Variable** | `PYTHON_VERSION = 3.11.9` |

---

## 🔮 Future Upgrades

Here's what I plan to add to make this project more powerful:

- [ ] 🤖 **Upgrade to BERT / Transformer models** for higher accuracy (90%+)
- [ ] 📊 **Multi-class sentiment** — Very Positive / Neutral / Negative / Very Negative
- [ ] 🌐 **Multi-language support** — Tamil, Hindi, Arabic, French
- [ ] 🔌 **REST API** — Build a Flask/FastAPI backend so other apps can integrate this model
- [ ] 📡 **Real-time Twitter/X sentiment dashboard** — Live feed analysis
- [ ] 🏷️ **Domain-specific models** — Train separate models for medical, finance, food reviews
- [ ] 📈 **Confidence score display** — Show how certain the model is (e.g. 92% Positive)
- [ ] 🗄️ **Database integration** — Save and track all predictions over time
- [ ] 📱 **Mobile-friendly UI** — Improve the Streamlit layout for mobile users
- [ ] ☁️ **Upgrade hosting** — Move to AWS / GCP for faster cold starts and better reliability

---

## 🔗 Where This Can Be Integrated

This model can be plugged into many real-world systems:

-  **CRM tools** like Salesforce or HubSpot — tag leads by sentiment
-  **Customer support platforms** like Zendesk — auto-prioritize negative tickets
-  **E-commerce backends** — auto-moderate and flag negative product reviews
-  **Mobile apps** — via REST API integration
-  **BI Dashboards** like Power BI or Tableau — sentiment trend visualization
-  **Chatbots** — detect user frustration and escalate to human agents

---

##  Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Naveed**

- 🔗 GitHub: [@naveed8606](https://github.com/naveed8606)
- 💼 LinkedIn: [Connect with me](https://www.linkedin.com/in/)
- 🌐 Live App: [sentiment-analysis-spacy-1.onrender.com](https://sentiment-analysis-spacy-1.onrender.com)

---

---

⭐ **If you found this project useful, please give it a star on GitHub!** ⭐