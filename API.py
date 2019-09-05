import tweepy
import re
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pickle
import numpy as np
def sentence_stemming(sentence):
    tokens=word_tokenize(sentence.lower())
    clean_tokens=tokens[:]
    for token in tokens:
        if token in stopwords.words('english')+[',','.']:
            clean_tokens.remove(token)
    stemmed_tokens=[PorterStemmer().stem(w) for w in clean_tokens]
    return ' '.join(stemmed_tokens)

def process(tweet):
    text = ' '.join(re.sub("(RT)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", tweet).split())
    return sentence_stemming(text)

class model:
    def __init__(self, vectorizer=None, classifier=None):
        self.vectorizer = vectorizer
        self.classifier = classifier
    def save(self, filename):
        import pickle
        pickle.dump(self, open(filename, 'wb'))
    def load(self, filename):
        import pickle
        loaded_model = pickle.load(open(filename, 'rb'))
        self.vectorizer = loaded_model.vectorizer
        self.classifier = loaded_model.classifier
    def predict(self, X):
        return self.classifier.predict(self.vectorizer.transform(X))



def calculate_sentiment(query,date_since= "2018-11-16",num=20):
    consumer_key = 'u3sfnSpirMhLvPRC37kvi03sw'
    consumer_secret = '90f5W6be4VmVlJMvHS7pqoC5orD4eQSoknPLM3jQDw91zhyZP8'
    access_key= '1135307851323400192-A19WMtpajWzRwclbTNwzdzDdidXw5P'
    access_secret = 'XDOtNCYZCTCXNkSqt5yaSCL3Ku5JFsx3TJSgp4A3tbJQi'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    search_words=query
    tweets = tweepy.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since).items(num)
    tweets=list(tweets)
    processed_text=[process(tweet.text) for tweet in tweets]
    model=pickle.load(open('sentiment_analyzer.model.0', 'rb'))
    result = model.classifier.predict_proba(model.vectorizer.transform(processed_text))
    print(processed_text)
    final_result={'positive':np.mean(result, axis=0)[1]}
    return query,final_result

