import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocess import preprocess
from src.train import train_model
from src.predict import predict

# Load data
with open("data/sample_data.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Preprocess
df['cleaned_text'] = df['text'].apply(preprocess)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# ------------------------
# Bag of Words Model
# ------------------------
model_bow, vec_bow = train_model(X_train, y_train, method="bow")
y_pred_bow = model_bow.predict(vec_bow.transform(X_test))

print("\n--- Bag of Words ---")
print("Accuracy:", accuracy_score(y_test, y_pred_bow))
print("Classification Report:\n", classification_report(y_test, y_pred_bow, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bow))


# ------------------------
# TF-IDF Model
# ------------------------
model_tfidf, vec_tfidf = train_model(X_train, y_train, method="tfidf")
y_pred_tfidf = model_tfidf.predict(vec_tfidf.transform(X_test))

print("\n--- TF-IDF ---")
print("Accuracy:", accuracy_score(y_test, y_pred_tfidf))
print("Classification Report:\n", classification_report(y_test, y_pred_tfidf, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tfidf))


# ------------------------
# Custom Test (use better model)
# ------------------------
print("\nCustom Test (TF-IDF):")
# print(predict("I really loved this!", model_tfidf, vec_tfidf, preprocess))
print(predict("It was an amazing day!", model_tfidf, vec_tfidf, preprocess))
print("\nCustom Test (BoW):")
print(predict("It was an amazing day!", model_bow, vec_bow, preprocess))