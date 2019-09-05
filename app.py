import flask
from flask import request
from API import model, calculate_sentiment

#clf=pickle.load(open('sentiment_analyzer.model.0', 'rb'))

app = flask.Flask(__name__)

@app.route("/", methods=["GET"])
def predict():
  print(request.args)
  if(request.args):
    x_input, predictions = calculate_sentiment(request.args['query'])
    print(x_input)
    return flask.render_template('predictor.html',query=x_input,prediction=predictions)
  else:
    print("no input")
    predictions = {'positive':0.5}
    return flask.render_template('predictor.html',query='',prediction=predictions)

if __name__=="__main__":
  #app.run(debug=True  )
  app.run(host='0.0.0.0')
  #app.run()
