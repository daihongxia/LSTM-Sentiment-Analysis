import tweepy
import numpy as np
from processing import process

def calculate_sentiment(query, 
                        mod, 
                        word_to_index,
                        maxLen,
                        date_since= "2018-11-16",
                        num=300):

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
    tweets=[tweet.full_text for tweet in tweets]
    #processed_text=[process(tweet.full_text,word_to_index,maxLen) for tweet in tweets]
    processed_text=process(tweets,word_to_index,maxLen)
    #mod=pickle.load(open('sentiment_analyzer.model.0', 'rb'))
    #result = mod.classifier.predict_proba(mod.vectorizer.transform(processed_text))
    result = mod.predict(processed_text)
    most_negatives = np.argsort(result[:,0])[:3]
    most_positives = np.argsort(result[:,0])[-3:]
    print(most_negatives)
    print(most_positives)
    result_tweets = []
    for i in most_negatives:
        result_tweets.append(tweets[i])
    for i in most_positives:
        result_tweets.append(tweets[i])
    #result_tweets = [tweet.text for tweet in tweets[most_negatives]]
    print(result_tweets)
    final_result={'positive':str(format(np.mean(result, axis=0)[0]*100, '.2f'))+'%'}
    return query,final_result,result_tweets

