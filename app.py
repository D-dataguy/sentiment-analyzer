import streamlit as st
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    with open("model_stats.json") as f:
        stats = json.load(f)
    analyzer = SentimentIntensityAnalyzer()
    return model, vectorizer, stats, analyzer

model, vectorizer, stats, vader = load_model()

st.title("💬 Sentiment Analyzer")
st.markdown("Analyze text sentiment using a **Logistic Regression + TF-IDF** model with VADER sentiment scores.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Text")
    
    example_texts = {
        "Select an example...": "",
        "Positive review": "I absolutely love this product! It exceeded all my expectations and the quality is outstanding.",
        "Negative review": "Terrible experience. The product broke after one day and customer support was completely unhelpful.",
        "Neutral statement": "The package arrived on time and the product works as described in the listing.",
    }
    
    selected = st.selectbox("Load an example", list(example_texts.keys()))
    
    default_text = example_texts[selected] if selected != "Select an example..." else ""
    user_text = st.text_area("Enter text to analyze:", value=default_text, height=180,
                              placeholder="Type or paste any text here...")

    analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    st.subheader("Results")

    if analyze_btn and user_text.strip():
        # ML model prediction
        vec = vectorizer.transform([user_text])
        ml_pred = model.predict(vec)[0]
        ml_proba = model.predict_proba(vec)[0]
        classes = model.classes_

        # VADER scores
        vader_scores = vader.polarity_scores(user_text)

        emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        color_map = {"positive": "#639922", "negative": "#e24b4a", "neutral": "#ef9f27"}

        st.markdown(f"### {emoji_map[ml_pred]} Sentiment: **{ml_pred.capitalize()}**")

        # Probability bars
        st.markdown("**Model Confidence:**")
        for cls, prob in sorted(zip(classes, ml_proba), key=lambda x: -x[1]):
            col_a, col_b = st.columns([3, 1])
            col_a.progress(float(prob), text=cls.capitalize())
            col_b.write(f"{prob:.1%}")

        st.divider()
        st.markdown("**VADER Compound Score:**")
        compound = vader_scores["compound"]
        vader_label = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
        st.metric("Compound Score", f"{compound:.3f}", delta=vader_label)

        fig, ax = plt.subplots(figsize=(5, 2.5))
        vader_cats = ["Positive", "Neutral", "Negative"]
        vader_vals = [vader_scores["pos"], vader_scores["neu"], vader_scores["neg"]]
        colors = ["#639922", "#ef9f27", "#e24b4a"]
        bars = ax.barh(vader_cats, vader_vals, color=colors, height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_title("VADER Component Scores")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, vader_vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{val:.2f}", va="center", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)

    elif analyze_btn:
        st.warning("Please enter some text first.")
    else:
        st.info("Enter text on the left and click **Analyze Sentiment**.")

        st.divider()
        st.subheader("Model Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{stats['metrics']['accuracy']:.1%}")
        m2.metric("F1 Positive", f"{stats['metrics']['f1_positive']:.3f}")
        m3.metric("F1 Negative", f"{stats['metrics']['f1_negative']:.3f}")

st.divider()
st.caption("Built with scikit-learn, VADER & Streamlit · TF-IDF + Logistic Regression")
