import re
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np


def sentence_stemming(sentence):
    tokens=word_tokenize(sentence.lower())
    clean_tokens=tokens[:]
    for token in tokens:
        if token in stopwords.words('english')+[',','.']:
            clean_tokens.remove(token)
    stemmed_tokens=[PorterStemmer().stem(w) for w in clean_tokens]
    return ' '.join(stemmed_tokens)

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]                                 # number of training examples

    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words =[i.lower() for i in X[i].split()]
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            if w not in word_to_index:
                continue
            X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
            j = j+1
            if j>=max_len:
                break
    return X_indices


def process(tweet,word_to_index,maxLen):
    text = [' '.join(re.sub("(RT)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", atweet).split()) for atweet in tweet]
    text = [sentence_stemming(atext) for atext in text]
    text = np.asarray(text)
    text= sentences_to_indices(text, word_to_index, maxLen)
    return text