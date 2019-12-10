import os
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

from tensorflow_scripts.utils import img_util, label_map_util
from tensorflow_scripts.utils.label_util import load_labels
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

# from flask_cors import CORS

app = Flask(__name__)
# Uncomment this line if you are making a Cross domain request
# CORS(app)

@app.route('/vera_species/classify/', methods=['POST'])
def image_classifier():
    server_url = "http://localhost:8501/v1/models/vera_species:predict"
    label_file = 'models/vera_species/vera_species_labels.txt'
    # TODO: Utilizar try/catch para registro de logs e garantir que as requisições vieram parametrizadas corretamente
    prediction = {}
    prediction['id'] = request.json['id']
    prediction['type'] = "species_classification"
    prediction['images'] = []

    # Obtém a imagem a partir do url path informado pelo cliente:
    for image_path in request.json['images']:
        # Converte o arquivo num float array
        formatted_json_input = img_util.classification_pre_process(image_path)

        # Call tensorflow server
        headers = {"content-type": "application/json"}
        print(f'\n\nMaking request to {server_url}...\n')
        server_response = requests.post(server_url, data=formatted_json_input, headers=headers)
        print(f'Request returned\n')
        print(server_response)

        # Decoding results from TensorFlow Serving server
        pred = json.loads(server_response.content.decode('utf-8'))

        predictions = np.squeeze(pred['predictions'][0])
        
        results = []
        top_k = predictions.argsort()[-5:][::-1]
        labels = load_labels(label_file)
        for i in top_k:
            label = labels[i]
            score = float(predictions[i])
            results.append('{label},{score}'.format(label=label,score=score))

        #Se a classificação ocorreu sem erros, inclui-la no objeto de retorno
        # image_name = os.path.basename(image_path)
        prediction['images'].append('{image_path}:{results}'.format(image_path=image_path,results=results))

    # Returning JSON response to the frontend
    return jsonify(prediction)


@app.route('/vera_poles_trees/detect/', methods=['POST'])
def object_detection():
    label_file = 'models/vera_poles_trees/vera_poles_trees_labels.pbtxt'
    num_classes = 2
    server_url = "http://localhost:8501/v1/models/vera_poles_trees:predict"
    output_image = 'tf_output.json'

    #Objeto de resposta:
    prediction = {}
    prediction['id'] = request.json['id']
    prediction['type'] = "poles_trees_detection"
    prediction['images'] = []

    for image_path in request.json['images']:
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
        output_dict = img_util.post_process(server_response, image_np.shape, label_file)
        print(f'Post-processing done!\n')

        # Save output on disk
        # print(f'\n\nSaving output to {output_image}\n\n')
        # with open(output_image, 'w+') as outfile:
        #     json.dump(output_dict, outfile)
        # print(f'Output saved!\n')

        prediction['images'].append('{image_path}:{results}'.format(image_path=image_path,results=output_dict))

    return prediction