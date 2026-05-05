from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import random
import re
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"{request.method} {request.url.path} - {process_time:.4f}s")
    return response

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

entities = {
    "butt networks": "butt_networks",
    "opinion nest": "opinion_nest",
    "opinion-nest": "opinion_nest"
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

def extract_entity(text):
    text = text.lower()
    for name, entity in entities.items():
        if name in text:
            return entity
    return None

def predict(text):
    text = clean_input(text)

    entity = extract_entity(text)

    if entity:
        if "what" in text or "tell" in text or "about" in text:
            intent = f"about_{entity}"
            if intent in response_dict:
                return random.choice(response_dict[intent]), 0.95

    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    top_idx = np.argsort(probs)[::-1]

    ml_intents = []
    for i in top_idx[:3]:
        if probs[i] > 0.25:
            ml_intents.append((model.classes_[i], float(probs[i])))

    keyword_intents = detect_keywords(text)

    score_map = {}

    for intent, score in ml_intents:
        score_map[intent] = score_map.get(intent, 0) + score

    for ki in keyword_intents:
        score_map[ki] = score_map.get(ki, 0) + 0.65

    if not score_map:
        return None, 0.0

    sorted_intents = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

    best_intent, best_score = sorted_intents[0]

    replies = []

    for intent, score in sorted_intents[:2]:
        if intent in response_dict:
            replies.append(random.choice(response_dict[intent]))

    return " | ".join(replies), best_score

def fallback_response(text):
    return f"I understand you're asking about '{text}', but I don't have enough specific training data for this. Can you rephrase or provide more details?"

class Request(BaseModel):
    message: str

@app.post("/chat")
def chat(req: Request):
    reply, conf = predict(req.message)

    if reply is None or conf < 0.45:
        return {
            "response": fallback_response(req.message),
            "confidence": round(conf, 2),
            "mode": "fallback"
        }

    return {
        "response": reply,
        "confidence": round(conf, 2),
        "mode": "ml"
    }