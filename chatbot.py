from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import random
import re
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

file_path = "ai_chatbot.csv"

data = pd.read_csv(file_path)

def clean(text):
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower().strip())

data["message"] = data["message"].apply(clean)

X = data["message"]
Y = data["intent"]

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, Y)

response_dict = {}

for _, row in data.iterrows():
    response_dict.setdefault(row["intent"], []).append(row["response"])

intent_keywords = {
    "greeting": ["hi","hello","hey"],
    "goodbye": ["bye","exit","quit"],
    "thanks": ["thanks","thank you"],
    "pricing": ["price","cost","subscription","fee"],
    "services": ["service","offer","provide"],
    "ai_info": ["ai","machine learning","automation","ml"],
    "about": ["who","what is","about"],
    "contact": ["contact","email","number"]
}

def detect_keyword_intent(text):
    matched = []
    for intent, keys in intent_keywords.items():
        for k in keys:
            if k in text:
                matched.append(intent)
                break
    return matched

def save_new_data(message, intent, response):
    df = pd.read_csv(file_path)
    new_row = pd.DataFrame([[message, intent, response]], columns=["message","intent","response"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_path, index=False)

def retrain():
    global data, X, Y, vectorizer, model, response_dict

    data = pd.read_csv(file_path)
    data["message"] = data["message"].apply(clean)

    X = data["message"]
    Y = data["intent"]

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, Y)

    response_dict = {}
    for _, row in data.iterrows():
        response_dict.setdefault(row["intent"], []).append(row["response"])

def predict(text):
    text = clean(text)

    kw = detect_keyword_intent(text)

    if len(kw) == 1:
        intent = kw[0]
        if intent in response_dict:
            return random.choice(response_dict[intent]), 1.0, intent

    if len(kw) > 1:
        replies = []
        for i in kw:
            if i in response_dict:
                replies.append(random.choice(response_dict[i]))
        return " | ".join(replies), 1.0, "multi"

    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    idx = np.argmax(probs)

    intent = model.classes_[idx]
    conf = probs[idx]

    if intent in response_dict:
        return random.choice(response_dict[intent]), conf, intent

    return None, conf, "unknown"


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    return response


@app.get("/")
def home():
    return {"message": "Chatbot API is running"}


@app.post("/chat")
def chat(request: dict):
    user_input = request.get("message")

    reply, conf, intent = predict(user_input)

    if reply is None:
        reply = "I don't know this. Please teach me."
        return {
            "reply": reply,
            "intent": intent,
            "confidence": float(conf),
            "learning": False
        }

    return {
        "reply": reply,
        "intent": intent,
        "confidence": float(conf),
        "learning": True
    }


@app.post("/teach")
def teach(request: dict):
    message = clean(request.get("message"))
    intent = request.get("intent")
    response = request.get("response")

    save_new_data(message, intent, response)
    retrain()

    return {
        "message": "Learned successfully"
    }