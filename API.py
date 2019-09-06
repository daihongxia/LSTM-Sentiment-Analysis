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
from sklearn.feature_extraction.text import TfidfVectorizer

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


def calculate_sentiment(query, mod, date_since= "2018-11-16",num=300):
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
              since=date_since,
              tweet_mode='extended').items(num)
    tweets=list(tweets)
    processed_text=[process(tweet.full_text) for tweet in tweets]
    #mod=pickle.load(open('sentiment_analyzer.model.0', 'rb'))
    result = mod.classifier.predict_proba(mod.vectorizer.transform(processed_text))
    most_negatives = np.argsort(result[:,1])[:3]
    most_positives = np.argsort(result[:,0])[:3]
    print(most_negatives)
    print(most_positives)
    result_tweets = []
    for i in most_negatives:
        result_tweets.append(tweets[i].full_text)
    for i in most_positives:
        result_tweets.append(tweets[i].full_text)
    #result_tweets = [tweet.text for tweet in tweets[most_negatives]]
    print(result_tweets)
    final_result={'positive':str(format(np.mean(result, axis=0)[1]*100, '.2f'))+'%'}
    return query,final_result,result_tweets

