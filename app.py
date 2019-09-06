import flask
from flask import request
from API import calculate_sentiment
import ML_model
import pickle

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'model':
            from ML_model import model
            return model
        return super().find_class(module, name)

mod = CustomUnpickler(open('sentiment_analyzer.model.0', 'rb')).load()

app = flask.Flask(__name__)

@app.route("/", methods=["GET"])
def predict():
  print(request.args)
  if(request.args):
    x_input, predictions, result_tweets = calculate_sentiment(request.args['query'], mod)
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
