import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
import json
import re

print("Generating labeled sentiment dataset...")
np.random.seed(42)

positive_templates = [
    "I absolutely love this product, it's amazing!",
    "Fantastic experience, highly recommend to everyone.",
    "Great quality, exceeded my expectations completely.",
    "The service was excellent and very professional.",
    "Very happy with my purchase, will buy again.",
    "Outstanding performance, really impressed by the results.",
    "Best decision I ever made, totally worth it.",
    "Wonderful product, five stars without hesitation.",
    "Superb quality and fast delivery, very satisfied.",
    "Brilliant work, the team was incredibly helpful.",
    "Loved every aspect of this, truly remarkable.",
    "Incredible value for money, exceeded all expectations.",
    "So pleased with the outcome, absolutely brilliant.",
    "Exceptional quality, could not be more satisfied.",
    "Delightful experience from start to finish.",
]
negative_templates = [
    "This product is terrible, complete waste of money.",
    "Very disappointed, does not work as advertised.",
    "Worst purchase ever, broken after one day.",
    "Poor quality and bad customer service.",
    "Totally useless, returned it immediately.",
    "Do not buy this, complete garbage.",
    "Horrible experience, would not recommend to anyone.",
    "Failed within a week, extremely frustrated.",
    "Cheap materials, fell apart instantly.",
    "Awful product, nothing like the description.",
    "Very bad quality, stopped working immediately.",
    "Wasted my money on this junk.",
    "Terrible service, nobody helped me at all.",
    "Broken on arrival, extremely disappointed.",
    "Disgusting quality, absolutely the worst.",
]
neutral_templates = [
    "The product arrived on time and works as expected.",
    "It is okay, nothing special about it.",
    "Does the job, but there are better options available.",
    "Average product, not bad but not great either.",
    "Received the item, seems to work fine so far.",
    "Standard product, does what it says.",
    "It is acceptable for the price paid.",
    "Decent quality, met my basic requirements.",
    "Works as described, no complaints.",
    "Ordinary product, gets the job done.",
    "Nothing remarkable about it but it works.",
    "Fair product, matches the description.",
    "Reasonable quality for the price.",
    "It serves its purpose adequately.",
    "Meets expectations, neither great nor bad.",
]

def augment(template, n):
    words = template.split()
    results = [template]
    for _ in range(n - 1):
        sample = words.copy()
        if len(sample) > 4:
            i = np.random.randint(1, len(sample) - 1)
            sample.pop(i)
        results.append(" ".join(sample))
    return results

texts, labels = [], []
for tmpl in positive_templates:
    for t in augment(tmpl, 30):
        texts.append(t); labels.append("positive")
for tmpl in negative_templates:
    for t in augment(tmpl, 30):
        texts.append(t); labels.append("negative")
for tmpl in neutral_templates:
    for t in augment(tmpl, 20):
        texts.append(t); labels.append("neutral")

df = pd.DataFrame({"text": texts, "label": labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("sentiment_data.csv", index=False)
print(f"Dataset: {len(df)} samples — {df['label'].value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training Logistic Regression classifier...")
model = LogisticRegression(max_iter=500, random_state=42, C=1.0)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"]).tolist()

print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

stats = {
    "metrics": {
        "accuracy": round(acc, 4),
        "f1_positive": round(report["positive"]["f1-score"], 4),
        "f1_negative": round(report["negative"]["f1-score"], 4),
        "f1_neutral": round(report.get("neutral", {}).get("f1-score", 0), 4),
    },
    "confusion_matrix": cm,
    "classes": ["positive", "neutral", "negative"]
}
with open("model_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("Saved: model.pkl, vectorizer.pkl, model_stats.json, sentiment_data.csv")
