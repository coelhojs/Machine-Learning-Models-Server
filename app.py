import base64
import json
from io import BytesIO

import numpy as np
import requests
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image

from tensorflow_scripts.image_classification.img_utils import pre_process

# from flask_cors import CORS

app = Flask(__name__)

# Uncomment this line if you are making a Cross domain request
# CORS(app)

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


@app.route('/vera_species/classify/', methods=['POST'])
def image_classifier():

    # Obtém a imagem a partir do url path informado pelo cliente:
    # Converte o arquivo num float array
    response = request.json['data']
    formatted_json_input = pre_process(response)

    # TODO: Verificar se essa linha é necessária
    # this line is added because of a bug in tf_serving(1.10.0-dev)
    #img = img.astype('float16')

    # Making POST request
    headers = {"content-type": "application/json"}
    response = requests.post(
        'http://localhost:8501/v1/models/vera_species:predict', headers=headers, data=formatted_json_input)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(response.content.decode('utf-8'))

    # Returning JSON response to the frontend
    return jsonify(inception_v3.decode_predictions(np.array(pred['predictions']))[0])


# @app.route('/vera_species/retrain/', methods=['POST'])
