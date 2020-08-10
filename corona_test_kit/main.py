from flask import Flask
app = Flask(__name__)

import pickle

file = open(model.pkl)
clf = pickle.load(file)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == "__main__":
    app.run(debug=True)