from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train, method="bow"):
    if method == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=(1,2))
    else:
        vectorizer = CountVectorizer(ngram_range=(1,2))

    X_train_vec = vectorizer.fit_transform(X_train)

    # model = LogisticRegression(max_iter=200)
    model = LogisticRegression(max_iter=200, class_weight='balanced')
    model.fit(X_train_vec, y_train)

    return model, vectorizer