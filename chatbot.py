from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


data = pd.read_csv("ai_chatbot.csv")

# Features & Labels
X = data["message"]
Y = data["intent"]

# Convert text → numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)   

# Train model
model = LogisticRegression()
model.fit(X_vec, Y)

response_dict = {}
for _, row in data.iterrows():
    response_dict[row["intent"]] = row["response"]


def chatbot():
    print("🤖 AI Powered Chatbot (type 'exit' to end):")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break

        input_vec = vectorizer.transform([user_input])

        intent = model.predict(input_vec)[0]

        reply = response_dict.get(intent, "Sorry, I don't understand")

        print("Bot:", reply)

chatbot()