import base64
import json
from io import BytesIO

import numpy as np
import requests
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image

from tensorflow_scripts.utils.label_util import load_labels
from tensorflow_scripts.utils.img_util import pre_process

# from flask_cors import CORS

app = Flask(__name__)
label_file = 'tensorflow_serving/vera_species/1/labels.txt'

# Uncomment this line if you are making a Cross domain request
# CORS(app)

@app.route('/vera_species/classify/', methods=['POST'])
def image_classifier():
    # Obtém a imagem a partir do url path informado pelo cliente:
    # Converte o arquivo num float array
    response = request.json['data']
    formatted_json_input = pre_process(response)

    # TODO: Verificar se essa linha é necessária
    # this line is added because of a bug in tf_serving(1.10.0-dev)

    # Making POST request
    headers = {"content-type": "application/json"}
    response = requests.post(
        'http://localhost:8501/v1/models/vera_species:predict', headers=headers, data=formatted_json_input)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(response.content.decode('utf-8'))

    predictions = np.squeeze(pred['predictions'][0])

    results = []
    top_k = predictions.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        label = labels[i]
        score = float(predictions[i])
        results.append('{label},{score}'.format(label=label,score=score))

    # Returning JSON response to the frontend
    return jsonify(results)


@app.route('/vera_species/classify/', methods=['POST'])
def image_classifier():
    # Obtém a imagem a partir do url path informado pelo cliente:
    # Converte o arquivo num float array
    response = request.json['data']
    formatted_json_input = pre_process(response)

    # TODO: Verificar se essa linha é necessária
    # this line is added because of a bug in tf_serving(1.10.0-dev)

    # Making POST request
    headers = {"content-type": "application/json"}
    response = requests.post(
        'http://localhost:8501/v1/models/vera_species:predict', headers=headers, data=formatted_json_input)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(response.content.decode('utf-8'))

    predictions = np.squeeze(pred['predictions'][0])

    results = []
    top_k = predictions.argsort()[-5:][::-1]
    labels = load_labels(label_file)
    for i in top_k:
        label = labels[i]
        score = float(predictions[i])
        results.append('{label},{score}'.format(label=label,score=score))

    # Returning JSON response to the frontend
    return jsonify(results)


# @app.route('/vera_species/retrain/', methods=['POST'])
