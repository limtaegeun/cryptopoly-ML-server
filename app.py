# USAGE
# Start the server:
# 	python app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submit a request via Python:
#	python simple_request.py
from flask import Flask, request, jsonify
import numpy as np
import io
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import json
from keras.models import model_from_json
import joblib
# import trainModel

app = Flask(__name__)
model = None
scaler = None

def load_model(dir):
    # load predict model
    global model
    json_file = open(dir + "/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights(dir + "/model.h5")
    print("Loaded model from disk")
    print("model : " + dir)

def load_scaler(dir):
    global scaler
    scaler = joblib.load(dir + '/scaler.plk')
    print("Loaded scaler from disk")

def prepare_data(data_array):
    global scaler
    return scaler.transform(data_array)

def origin_data(data_array):
    global scaler
    return scaler.inverse_transform(data_array)

@app.route("/train", methods=["GET"])
def train():
    if request.method == "GET":
        base = request.args.get('base')
        quote = request.args.get('quote')
        pair = request.args.get('pair')
        start = request.args.get('start')
        end = request.args.get('end')
        period = request.args.get('period')
        print(pair)
        trainModel.train(pair, start, end, period)
        return 'success'

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        jsonData = json.loads(request.form['data'])
        print(jsonData)
        nparray = np.asarray(jsonData)
        prepared_data = prepare_data(nparray)
        reshaped_data = np.reshape(prepared_data, ( prepared_data.shape[0], prepared_data.shape[1],1))
        print(reshaped_data)
        predicted = model.predict(reshaped_data)
        originData = origin_data(predicted)
        print(originData)
        return jsonify(originData.tolist())




@app.route('/')
def hello_world():
    return 'Hello World!'


print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
load_model("USDT_BTC/86400")
load_scaler("USDT_BTC/86400")
if __name__ == '__main__':
    app.run()

