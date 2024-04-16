from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from emot.emo_unicode import UNICODE_EMOJI
from nltk.stem import WordNetLemmatizer
from preprocessing import LemmaTokenizer

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words=set(stop_words)
print(stop_words)

emoji = list(UNICODE_EMOJI.keys())

app = Flask(__name__)

# Load the sentiment analysis model and the vectorizer
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)


def clean_tokens(text):
    # 1 create tokens
    tokens = word_tokenize(text)
    # 2 lower case
    tokens = [w.lower() for w in tokens]
    # 3 remove punctuations
    stripped = [word for word in tokens if word.isalpha()]
    # 4 remove stop_words
    stop_words = set(stopwords.words('english'))
    words = [w for w in stripped if not w in stop_words]
    # remove emojis
    no_emoji = [w for w in words if not w in emoji]
    # join cleaned tokens into a single string
    clean_text = ' '.join(no_emoji)
    # return the cleaned string
    return clean_text

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')

        # Preprocess the comment
        preprocessed_comment = clean_tokens(comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        # Predict the sentiment
        sentiment = clf.predict(comment_vector)[0]

        return render_template('index.html', sentiment=sentiment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)