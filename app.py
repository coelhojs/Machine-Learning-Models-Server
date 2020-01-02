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
    #Tenta conexão com Tensorflow Serving, se não conseguir conectar, usar script local:
    try:
        prediction = {}
        prediction['RequestId'] = request.json['Id']
        prediction['type'] = "Species_classification"
        prediction['Detections'] = []
        
        #test_request
        requests.get("http://localhost:8501/v1/models/vera_species")

        label_file = 'models/vera_species/vera_species_labels.txt'
        server_url = "http://localhost:8501/v1/models/vera_species:predict"

        print("Tensorflow Serving detectado. Utilizando para classificação")

        # Obtém a imagem a partir do url path informado pelo cliente:
        for image_path in request.json['Images']:

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
        prediction['Detections'].append('{image_path}:{results}'.format(image_path=image_path,results=results))

    except:
        print("Tensorflow Serving não detectado. Utilizando scripts locais")

        label_file = 'C:/Machine-Learning-Models-Server/models_inference/vera_species/vera_species_labels.txt'
        model_path = 'C:/Machine-Learning-Models-Server/models_inference/vera_species/1/retrained_graph.pb'
        prediction['Detections'] = image_classifier(request.json['Images'], model_path, label_file)

    # Returning JSON response to the frontend
    return jsonify(prediction)


@app.route('/vera_poles_trees/detect/', methods=['POST'])
def object_detection():
    
    try:
        #Objeto de resposta:
        prediction = {}
        prediction['RequestId'] = request.json['Id']
        prediction['type'] = "poles_trees_detection"
        prediction['Detections'] = []

        #handshake?
        requests.get("http://localhost:8501/v1/models/vera_poles_trees")
        
        label_file = 'models/vera_poles_trees/vera_poles_trees_labels.pbtxt'
        server_url = "http://localhost:8501/v1/models/vera_poles_trees:predict"

        for image_path in request.json['Images']:
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

            # Formatando resultado para o modelo esperado pelo Vera
            inference_dict = {}
            inference_dict['ImagePath'] = image_path
            inference_dict['Class'] = output_dict['detection_classes']
            inference_dict['BoundingBoxes'] = output_dict['detection_boxes']
            inference_dict['Score'] = np.array(output_dict['detection_scores']).tolist()
            inference_dict['NumDetections'] = output_dict['num_detections']

            print(f'Post-processing done!\n')


            prediction['Detections'].append(inference_dict)
    
        return jsonify(prediction)

    except:
        print("Tensorflow Serving não detectado. Utilizando scripts locais")

        label_file = 'C:/Machine-Learning-Models-Server/models_inference/vera_poles_trees/vera_poles_trees_labels.pbtxt'
        model_path = 'C:/Machine-Learning-Models-Server/models_inference/vera_poles_trees/1/frozen_inference_graph.pb'
        
        #imagesList = request.json['Images']
		
        #validated_req = validate_paths(imagesList)
		
        #request.json['Images'] = validated_req
		
        images_predictions = objects_detector(validate_paths(request.json['Images'], request.remote_addr), model_path, img_util.load_labels(label_file))

        for x in range(len(images_predictions)):
            prediction['Detections'].append(images_predictions[x])


        return jsonify(prediction)

def validate_paths(images, remote_addr):
    newList = []
    for imagepath in images:
        if ("C:" in imagepath):
            newPath = imagepath.replace(
                "C:/", "//{remote_addr}/c/".format(remote_addr=remote_addr))
        elif ("c:" in imagepath):
            newPath = imagepath.replace(
                "c:/", "//{remote_addr}/c/".format(remote_addr=remote_addr))
        newList.append(newPath)
    return newList