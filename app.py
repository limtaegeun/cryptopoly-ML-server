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
model = {}
scaler = {}

def load_model(dir, name):
    # load predict model
    global model
    json_file = open(dir + "/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model[name] = model_from_json(loaded_model_json)

    model[name].load_weights(dir + "/model.h5")
    print("Loaded model from disk : " + name)

def load_scaler(dir, name):
    global scaler
    scaler[name] = joblib.load(dir + '/scaler.plk')
    print("Loaded scaler from disk :" + name)

def prepare_data(data_array, modelName):
    global scaler
    return scaler[modelName].transform(data_array)

def origin_data(data_array, modelName):
    global scaler
    return scaler[modelName].inverse_transform(data_array)

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
        modelName = request.form['pair']
        print(jsonData)
        nparray = np.asarray(jsonData)
        print(nparray)
        prepared_data = prepare_data(nparray, modelName)
        reshaped_data = np.reshape(prepared_data, ( 1, prepared_data.shape[0], prepared_data.shape[1]))
        print(reshaped_data)
        predicted = model[modelName].predict(reshaped_data)
        print('before origin')
        print(predicted)
        originData = origin_data(predicted, modelName)
        print('predict : ' + modelName)
        print(originData)
        return jsonify(originData.tolist())


@app.route('/')
def hello_world():
    return 'Hello World!'


print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
load_model("models/USDT_BTC/86400", 'USDT_BTC')
load_scaler("models/USDT_BTC/86400", 'USDT_BTC')
load_model("models/USDT_ETH/86400", 'USDT_ETH')
load_scaler("models/USDT_ETH/86400", 'USDT_ETH')
load_model("models/USDT_XRP/86400", 'USDT_XRP')
load_scaler("models/USDT_XRP/86400", 'USDT_XRP')
if __name__ == '__main__':
    app.run()

