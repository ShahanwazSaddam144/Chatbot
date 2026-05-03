from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import random
import re
import numpy as np

app = FastAPI()

data = pd.read_csv("ai_chatbot.csv")

def clean(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text

data["message"] = data["message"].apply(clean)

X = data["message"]
Y = data["intent"]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, Y)

response_dict = {}

for _, row in data.iterrows():
    response_dict.setdefault(row["intent"], []).append(row["response"])

intent_keywords = {
    "pricing": ["price", "pricing", "cost", "plan", "subscription", "fee"],
    "services": ["service", "offer", "provide", "web", "app"],
    "ai_info": ["ai", "machine learning", "automation", "ml"],
    "about": ["who", "what is", "about", "company", "software house"],
    "greeting": ["hi", "hello", "hey", "good morning", "good evening"],
    "goodbye": ["bye", "exit", "see you", "goodbye"],
    "support": ["support", "help", "error", "bug", "issue"],
    "security": ["safe", "security", "privacy", "data"],
    "automation": ["automation", "workflow", "connect", "task"],
    "contact": ["contact", "info", "email", "number"]
}

def clean_input(text):
    return clean(text)

def detect_keywords(text):
    found = []
    for intent, keys in intent_keywords.items():
        for k in keys:
            if k in text:
                found.append(intent)
                break
    return found

def predict(text):
    text = clean_input(text)
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    top_idx = np.argsort(probs)[::-1]
    intents = []
    for i in top_idx[:3]:
        if probs[i] > 0.20:
            intents.append((model.classes_[i], probs[i]))
    keyword_intents = detect_keywords(text)
    final_intents = []
    for i, c in intents:
        final_intents.append(i)
    for ki in keyword_intents:
        if ki not in final_intents:
            final_intents.append(ki)
    if not final_intents:
        return "Sorry, I didn't understand that.", 0.0
    replies = []
    best_conf = 0
    for intent in final_intents:
        if intent in response_dict:
            replies.append(random.choice(response_dict[intent]))
            best_conf = max(best_conf, max(probs))
    return " | ".join(replies), best_conf

class Request(BaseModel):
    message: str

@app.post("/chat")
def chat(req: Request):
    reply, conf = predict(req.message)
    return {"response": reply, "confidence": round(conf, 2)}