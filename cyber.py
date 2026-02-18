# cyberbully_app.py

import subprocess
import sys

# ----------------------------
# 0. Install required packages if missing
# ----------------------------
required_packages = ["pandas", "scikit-learn", "nltk", "ollama", "gradio", "numpy"]
for pkg in required_packages:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ----------------------------
# 1. Imports
# ----------------------------
import pandas as pd
import nltk
import string
import ollama
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import gradio as gr

# ----------------------------
# 2. Download stopwords
# ----------------------------
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# ----------------------------
# 3. Text preprocessing
# ----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ----------------------------
# 4. Load dataset
# ----------------------------
df = pd.read_csv("cyberbullying_tweets.csv")
df['clean_text'] = df['tweet_text'].astype(str).apply(preprocess)

X = df['clean_text']
y = df['cyberbullying_type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 5. Train SVM model
# ----------------------------
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', LinearSVC())
])

model.fit(X_train, y_train)

# ----------------------------
# 6. Calculate Model Accuracy
# ----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ----------------------------
# 7. Gradio prediction function
# ----------------------------
def detect_cyberbullying(sentence):
    if sentence.strip() == "":
        return "<div style='color:white; font-size:16px;'>Please enter a sentence.</div>"

    clean_input = preprocess(sentence)

    prediction = model.predict([clean_input])[0]
    decision_score = model.decision_function([clean_input])
    confidence = round(float(np.max(decision_score)), 3)

    # Ask Ollama for explanation
    prompt = f"""
    The sentence is: "{sentence}"
    It is classified as: {prediction}.
    Briefly explain why this is considered cyberbullying or not.
    """

    try:
        response = ollama.chat(
            model='llama3',
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = response['message']['content']
    except Exception:
        explanation = "Ollama explanation unavailable. Make sure Ollama is running."

    # DARK THEME OUTPUT
    styled_output = f"""
    <div style="
        background-color:#000000;
        border:3px solid #ffffff;
        border-radius:12px;
        padding:25px;
        font-family:Arial, sans-serif;
        color:#ffffff;
    ">

        <p style="font-weight:bold; font-size:18px;">Input Sentence:</p>
        <p style="font-size:16px; margin-bottom:15px;">{sentence}</p>

        <p style="font-weight:bold; font-size:18px;">Predicted Category:</p>
        <p style="font-size:16px; margin-bottom:15px;">{prediction}</p>

        <p style="font-weight:bold; font-size:18px;">Confidence Score:</p>
        <p style="font-size:16px; margin-bottom:15px;">{confidence}</p>

        <p style="font-weight:bold; font-size:18px;">Model Accuracy:</p>
        <p style="font-size:16px; margin-bottom:20px;">{round(accuracy*100,2)}%</p>

        <div style="
            background-color:#1a1a1a;
            padding:15px;
            border-radius:8px;
            border:1px solid #ffffff;
        ">
            <p style="font-weight:bold; font-size:16px;">Explanation (Ollama):</p>
            <p style="font-size:15px;">{explanation}</p>
        </div>

    </div>
    """

    return styled_output

# ----------------------------
# 8. Launch Gradio UI
# ----------------------------
iface = gr.Interface(
    fn=detect_cyberbullying,
    inputs=gr.Textbox(lines=3, placeholder="Enter a sentence..."),
    outputs=gr.HTML(),
    title="🚨 Cyberbullying Detection System",
    description="Enter a sentence to detect cyberbullying, view confidence score, model accuracy, and AI explanation."
)

iface.launch()
