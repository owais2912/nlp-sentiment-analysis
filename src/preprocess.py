import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

if 'not' in stop_words:
    stop_words.remove('not')

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    # words = [ps.stem(word) for word in words if word not in stop_words]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)