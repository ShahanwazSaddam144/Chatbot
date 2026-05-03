from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import random
import re
import numpy as np

import pandas as pd

data = pd.read_csv("ai_chatbot.csv")

data["message"] = data["message"].str.lower().str.strip()

X = data["message"]
Y = data["intent"]

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=300)
model.fit(X_vec, Y)

response_dict = {}

for _, row in data.iterrows():
    intent = row["intent"]
    response = row["response"]

    if intent not in response_dict:
        response_dict[intent] = []

    response_dict[intent].append(response)

print("🤖 Chatbot trained successfully!")

def is_valid_input(text_vec):
    return text_vec.sum() > 0.2

def get_reply(text):
    input_vec = vectorizer.transform([text])

    if not is_valid_input(input_vec):
        return None, 0.0, None

    probs = model.predict_proba(input_vec)
    confidence = max(probs[0])
    intent = model.classes_[probs.argmax()]

    reply = random.choice(response_dict[intent])

    return reply, confidence, intent


def chatbot():
    print("\n🤖 AI Chatbot Ready (type exit to stop)\n")

    while True:
        user_input = input("You: ").lower().strip()

        if user_input == "exit":
            print("Bot: Goodbye! 👋")
            break

        parts = re.split(r"[?.!]| and ", user_input)

        for part in parts:
            part = part.strip()

            if not part:
                continue

            reply, confidence, intent = get_reply(part)

            # 🔥 FINAL FIX LOGIC
            if reply is None or confidence < 0.10:
                print("Bot: I didn’t understand that. Please ask something related to Butt Networks services.")
            else:
                print("Bot:", reply)


chatbot()