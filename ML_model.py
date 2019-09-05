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
