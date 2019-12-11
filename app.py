import base64
import json
import os
import time
from io import BytesIO
from os import listdir
from os.path import isfile, join

import numpy as np
import requests
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.contrib import util as contrib_util
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image

from tensorflow_scripts.image_classification.label_image import image_classifier
from tensorflow_scripts.object_detection.object_detection import objects_detector
from tensorflow_scripts.utils import img_util, label_map_util
from tensorflow_scripts.utils.label_util import load_labels

app = Flask(__name__)

@app.route('/vera_species/classify/', methods=['POST'])
def species_classifier():
    # TODO: Utilizar try/catch para registro de logs e garantir que as requisições vieram parametrizadas corretamente
    prediction = {}
    prediction['id'] = request.json['id']
    prediction['type'] = "species_classification"
    prediction['images'] = []

    #Tenta conexão com Tensorflow Serving, se não conseguir conectar, usar script local:
    try:
        
        #test_request
        requests.get("http://localhost:8501/v1/models/vera_species")

        label_file = 'models/vera_species/vera_species_labels.txt'
        server_url = "http://localhost:8501/v1/models/vera_species:predict"

        print("Tensorflow Serving detectado. Utilizando para classificação")

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
        prediction['images'].append('{image_path}:{results}'.format(image_path=image_path,results=results))

    except:
        print("Tensorflow Serving não detectado. Utilizando scripts locais para classificação")

        label_file = 'C:/Machine-Learning-Models-Server/models_inference/vera_species/vera_species_labels.txt'
        model_path = 'C:/Machine-Learning-Models-Server/models_inference/vera_species/1/retrained_graph.pb'
        prediction['images'] = image_classifier(request.json['images'], model_path, label_file)

    # Returning JSON response to the frontend
    return jsonify(prediction)


@app.route('/vera_poles_trees/detect/', methods=['POST'])
def object_detection():

    #Objeto de resposta:
    prediction = {}
    prediction['id'] = request.json['id']
    prediction['type'] = "poles_trees_detection"
    prediction['images'] = []

    try:
        #handshake?
        requests.get("http://localhost:8501/v1/models/vera_poles_trees")
        
        label_file = 'models/vera_poles_trees/vera_poles_trees_labels.pbtxt'
        server_url = "http://localhost:8501/v1/models/vera_poles_trees:predict"

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
            output_dict = img_util.post_process(server_response, image_np.shape, img_util.load_labels(label_file))
            print(f'Post-processing done!\n')

            prediction['images'].append('{image_path}:{results}'.format(image_path=image_path,results=output_dict))
    
        return jsonify(prediction)

    except:
        print("Tensorflow Serving não detectado. Utilizando scripts locais para classificação")

        label_file = 'C:/Machine-Learning-Models-Server/models_inference/vera_poles_trees/vera_poles_trees_labels.pbtxt'
        model_path = 'C:/Machine-Learning-Models-Server/models_inference/vera_poles_trees/1/frozen_inference_graph.pb'
        
        images_predictions = objects_detector(request.json['images'], model_path, img_util.load_labels(label_file))

        for x in range(len(images_predictions)):
            prediction['images'].append('{image_path}:{results}'.format(image_path=request.json['images'][x],results=images_predictions[x]))

        return jsonify(prediction)
