import flask
from flask import request
from src.API import calculate_sentiment
from src.processing import read_glove_vecs

import h5py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import model_from_json
import tensorflow as tf

PATH = './model/'
#mod = CustomUnpickler(open('sentiment_analyzer.model.0', 'rb')).load()
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(PATH+'glove.6B.50d.txt')
maxLen = 20

# load json and create model
json_file = open(PATH+'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(PATH+"model.h5")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

loaded_model._make_predict_function()
graph = tf.get_default_graph()

app = flask.Flask(__name__)

@app.route("/", methods=["GET"])
def predict():
    print(request.args)
    if(request.args):
        global graph
        with graph.as_default():
            x_input, predictions, result_tweets = calculate_sentiment(request.args['query'], loaded_model,word_to_index,maxLen)
        x_input = "The query you made, "+str(x_input)+", has a positive rate: "
        result_tweets = ['Some sample tweets we collected and analyzed:'] + result_tweets
        print(x_input)
        return flask.render_template('predictor.html',query=x_input,
                                 prediction=predictions,
                                 tweets=result_tweets)
    else:
        print("no input")
        predictions = {'positive':''}
        return flask.render_template('predictor.html',query='',
                                 prediction=predictions,
                                 tweets='')

if __name__=="__main__":
  #app.run(debug=True  )
  app.run(host='0.0.0.0')
  #app.run()
