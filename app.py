import base64
import json
import time
from io import BytesIO
from os import listdir
from os.path import isfile, join

# from grpc.beta import implementations
import grpc
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.contrib import util as contrib_util
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from tensorflow_scripts.utils import img_util, label_map_util
from tensorflow_scripts.utils.label_util import load_labels

# from flask_cors import CORS

app = Flask(__name__)
# Uncomment this line if you are making a Cross domain request
# CORS(app)

@app.route('/vera_species/classify/', methods=['POST'])
def image_classifier():
    label_file = 'tensorflow_serving/vera_species/1/labels.txt'
    # Obtém a imagem a partir do url path informado pelo cliente:
    # Converte o arquivo num float array
    response = request.json['data']
    formatted_json_input = img_util.classification_pre_process(response)

    # TODO: Verificar se essa linha é necessária
    # this line is added because of a bug in tf_serving(1.10.0-dev)

    # Making POST request
    headers = {"content-type": "application/json"}
    response = requests.post(
        'http:/localhost:8501/v1/models/vera_species:predict', headers=headers, data=formatted_json_input)

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


@app.route('/vera_poles_trees/detect/', methods=['POST'])
def object_detection():

    # Map args to var
    image_path = request.json['data']
    server_url = "http://localhost:8501/v1/models/vera_poles_trees:predict"
    output_image = 'tf_output.json'
    path_to_labels = 'tensorflow_serving/vera_poles_trees/1/label_map.pbtxt'

    # Build input data
    print(f'\n\nPre-processing input file {image_path}...\n')
    formatted_json_input = img_util.object_detection_pre_process(image_path)
    print('Pre-processing done! \n')

    # Call tensorflow server
    headers = {"content-type": "application/json"}
    print(f'\n\nMaking request to {server_url}...\n')
    server_response = requests.post(server_url, data=formatted_json_input, headers=headers)
    print(f'Request returned\n')
    print(server_response)

    # Post process output
    print(f'\n\nPost-processing server response...\n')
    image = Image.open(image_path).convert("RGB")
    image_np = img_util.load_image_into_numpy_array(image)
    output_dict = img_util.post_process(server_response, image_np.shape)
    print(f'Post-processing done!\n')

    # Save output on disk
    print(f'\n\nSaving output to {output_image}\n\n')
    with open(output_image, 'w+') as outfile:
        json.dump(json.loads(output_dict), outfile)
    print(f'Output saved!\n')

    return output_dict

# @app.route('/vera_species/retrain/', methods=['POST'])
