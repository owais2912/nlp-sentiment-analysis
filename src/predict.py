def predict(text, model, vectorizer, preprocess_fn):
    cleaned = preprocess_fn(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "Positive" if prediction == 1 else "Negative"