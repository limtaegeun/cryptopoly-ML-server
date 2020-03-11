# USAGE
# Start the server:
# 	python app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submit a request via Python:
#	python simple_request.py
import flask
import numpy as np
import io
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
from sklearn.externals import joblib

app = flask.Flask(__name__)
model = None
scaler = None
def load_model():
    # load predict model
    global model
    json_file = open("model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights("model.h5")
    print("Loaded model from disk")

def load_scaler():
    global scaler
    scaler = joblib.load('scaler.plk')

def prepare_data(data_array):
    global scaler
    return scaler.transform(data_array)

@app.route("/predict", methods=["POST"])
def predict():
    # if flask.request.method == "POST":
        # if flask.request.body:
        #     prepared_data = prepare_data()
    return 'hi'


if __name__ == '__main__':
    app.run()
