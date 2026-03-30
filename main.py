import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.preprocess import preprocess
from src.train import train_model
from src.predict import predict

# Sample dataset
data = {
    "text": [
        "I love this product",
        "This is the worst thing ever",
        "Absolutely amazing experience",
        "I hate this",
        "Not good at all",
        "Really happy with the service",
        "Terrible support",
        "Very satisfied",
        "Bad quality",
        "Excellent performance",
        "I hated the experience"
    ],
    "label": [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Preprocess text
df['cleaned_text'] = df['text'].apply(preprocess)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'], test_size=0.2, random_state=42
)

# Train model
model, vectorizer = train_model(X_train, y_train)

# Transform test data
X_test_vec = vectorizer.transform(X_test)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Custom test
print("\nCustom Test:")
print(predict("I really hated this!", model, vectorizer, preprocess))