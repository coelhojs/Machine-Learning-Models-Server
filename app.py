import asyncio
import base64
import json
import os
import time
from io import BytesIO
from os import listdir
from os.path import isfile, join
from threading import Thread

import numpy as np
import requests
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.contrib import util as contrib_util
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image

from object_detection_caller import (object_detection_batch_script,
                                     object_detection_batch_serving)
from tensorflow_scripts.image_classification.label_image import \
    image_classifier
from tensorflow_scripts.object_detection.object_detection import \
    objects_detector
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


@app.route('/vera_poles_trees/detect_simple/', methods=['POST'])
def object_detection_simple():

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
            inference_dict['Classes'] = output_dict['detection_classes']
            inference_dict['BoundingBoxes'] = output_dict['detection_boxes']
            inference_dict['Scores'] = np.array(output_dict['detection_scores']).tolist()
            inference_dict['NumDetections'] = output_dict['num_detections']

            print(f'Post-processing done!\n')


            prediction['Detections'].append(inference_dict)
    
        return jsonify(prediction)

    except:
        print("Tensorflow Serving não detectado. Utilizando scripts locais")

        label_file = 'C:/Machine-Learning-Models-Server/models_inference/vera_poles_trees/vera_poles_trees_labels.pbtxt'
        model_path = 'C:/Machine-Learning-Models-Server/models_inference/vera_poles_trees/1/frozen_inference_graph.pb'
        
        images_predictions = objects_detector(request.json['Images'], model_path, img_util.load_labels(label_file))

        for x in range(len(images_predictions)):
            prediction['Detections'].append(images_predictions[x])


        return jsonify(prediction)


@app.route('/vera_poles_trees/detect_batch/', methods=['POST'])
def object_detection_batch():

    labels = img_util.load_labels('models/vera_poles_trees/vera_poles_trees_labels.pbtxt')
    model_inference = 'C:/Machine-Learning-Models-Server/models_inference/vera_poles_trees/1/frozen_inference_graph.pb'
    server_url = "http://localhost:8501/v1/models/vera_poles_trees:predict"

    #Quebra a lista de imagens em sublistas de determinado tamanho:
    list_size = 50
            
    splitted_list_images = [request.json['Images'][i * list_size:(i + 1) * list_size] for i in range((len(request.json['Images']) + list_size - 1) // list_size )]  

    #Objeto de resposta:
    prediction = {}
    prediction['RequestId'] = request.json['Id']
    prediction['Type'] = "poles_trees_detection"
    prediction['UserToken'] = request.json['UserToken']
    
    try:
        #handshake?
        requests.get("http://localhost:8501/v1/models/vera_poles_trees")
      
        task = Thread(target=object_detection_batch_serving(prediction_obj, splitted_list_images, model_path, labels, server_url))

    except:
        print("Tensorflow Serving não detectado. Utilizando scripts locais")

        task = Thread(target=object_detection_batch_script(prediction, splitted_list_images, model_inference, labels))

    finally:
        task.daemon = True
        task.start()
        return "Detecção em lote iniciada"
