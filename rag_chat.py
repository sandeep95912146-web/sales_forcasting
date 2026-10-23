import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------
# Simple RAG: Retrieve the most relevant explanation, then Answer
# ---------------------------------------------------------------

# 1. Load the SHAP explanations saved by explain_shap.py
df = pd.read_csv("shap_values.csv", parse_dates=["date"])
drivers = [c for c in df.columns if c not in ("date", "forecast", "actual")]


def describe_day(row):
    items = sorted(((c, row[c]) for c in drivers), key=lambda kv: -abs(kv[1]))
    ups = ", ".join(f"{c} (+{v:.0f})" for c, v in items if v > 0) or "nothing"
    downs = ", ".join(f"{c} ({v:.0f})" for c, v in items if v < 0) or "nothing"
    day = row["date"]
    return (
        f"On {day.date()} ({day.day_name()}, {day.month_name()} {day.year}) "
        f"the forecast was {row['forecast']:.1f} and the actual sales were {row['actual']:.1f}. "
        f"Factors that pushed sales up: {ups}. "
        f"Factors that pushed sales down: {downs}."
    )


# 2. Build the knowledge base (one document per explained day + overall summary)
docs = [describe_day(r) for _, r in df.iterrows()]
imp = df[drivers].abs().mean().sort_values(ascending=False)
summary = (
    f"The factor that matters most, the biggest driver and main factor of the sales forecast, is {imp.index[0]}. "
    "Overall, the most important factors for the sales forecast, from most to least important, are: "
    + ", ".join(f"{c} ({v:.1f})" for c, v in imp.items())
    + ". Importance is the average effect on the forecast in sales units."
)
method = (
    "How the model works: a linear model captures the trend, then an LSTM deep learning model "
    "learns the remaining patterns such as seasonality. SHAP explains how much each factor "
    "(promo, holiday, school holiday, weekend, trend, season, and recent sales pattern) adds or removes from each forecast."
)
docs += [summary, method]

# 3. Index the documents (TF-IDF search)
vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(docs)


def find_date(question):
    m = re.search(r"\d{4}-\d{2}-\d{2}", question)
    if m:
        return pd.to_datetime(m.group(0), errors="coerce")
    m = re.search(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}", question)
    if m:
        return pd.to_datetime(m.group(0), errors="coerce")
    return pd.NaT


def retrieve(question, k=2):
    d = find_date(question)
    if pd.notna(d):
        hit = df.index[df["date"] == d]
        if len(hit):
            return [docs[hit[0]]], None
        nearest = df.iloc[(df["date"] - d).abs().argsort()[:3]]["date"].dt.date.tolist()
        return [], nearest
    sims = cosine_similarity(vectorizer.transform([question]), matrix)[0]
    top = sims.argsort()[::-1][:k]
    return [docs[i] for i in top if sims[i] > 0], None


def answer(question):
    found, nearest = retrieve(question)
    if nearest:
        return "I have no explanation for that date. Dates I can explain nearby: " + ", ".join(map(str, nearest))
    if not found:
        return "I could not find anything about that. Try asking about a date, or about promo, holiday, school holiday, weekend, trend or season."
    return " ".join(found)


# 4. Chat loop
if __name__ == "__main__":
    print("Sales forecast assistant. Ask a question, for example:")
    print("  Why was the forecast high on", df['date'].iloc[len(df) // 2].date(), "?")
    print("  Which factor matters most?")
    print("  How does the model work?")
    print("Type quit to stop.")
    print()
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print("Assistant:", answer(q))
            print()
