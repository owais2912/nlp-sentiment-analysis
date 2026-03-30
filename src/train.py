from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    cv = CountVectorizer()
    X_train_vec = cv.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    return model, cv