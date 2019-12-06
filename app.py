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
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image
from tensorflow.contrib import util as contrib_util
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from tensorflow_scripts.utils import label_map_util
from tensorflow_scripts.utils.img_util import (classification_pre_process,
                                               object_detection_pre_process)
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
    formatted_json_input = classification_pre_process(response)

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
    IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

    tf.app.flags.DEFINE_string('tf_server', 'localhost:8500',
                            'PredictionService host:port')
    tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.image:
        with open(FLAGS.image, 'rb') as f:
            data = f.read()
    else:
        # Download the image since we weren't given one
        dl_request = requests.get(IMAGE_URL, stream=True)
        dl_request.raise_for_status()
        data = dl_request.content

    channel = grpc.insecure_channel(FLAGS.tf_server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # Send request
    # See prediction_service.proto for gRPC request/response details.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'vera_poles_trees'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['inputs'].CopyFrom(
        contrib_util.make_tensor_proto(data, shape=[len(data)]))
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    print(result)

    return jsonify(result)

# @app.route('/vera_species/retrain/', methods=['POST'])
