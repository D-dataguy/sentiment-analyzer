# 💬 Sentiment Analyzer

A machine learning web app that classifies text as **Positive**, **Negative**, or **Neutral** using TF-IDF features and Logistic Regression, combined with VADER sentiment scoring.

## 🚀 Live Demo
> Deploy to [Streamlit Cloud](https://sentiment-analyzer-iyxkjwpguqzquogwxr97nx.streamlit.app/) for free.

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 100% (on synthetic data) |
| F1 Positive | 1.000 |
| F1 Negative | 1.000 |

## 🧠 How It Works
1. **TF-IDF Vectorization** — Converts text into numerical feature vectors (bigrams, 5,000 features)
2. **Logistic Regression** — Classifies sentiment into 3 classes
3. **VADER** — Provides a secondary rule-based sentiment score for comparison

## 🛠️ Tech Stack
- **Model**: Logistic Regression with TF-IDF (scikit-learn)
- **NLP**: VADER SentimentIntensityAnalyzer
- **Frontend**: Streamlit
- **Visualization**: Matplotlib

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analyzer
cd sentiment-analyzer
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## 📁 Project Structure
```
sentiment-analyzer/
├── app.py
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── model_stats.json
├── sentiment_data.csv
├── requirements.txt
└── README.md
```

---
*Built as part of an ML portfolio project.*
