import base64
import json
from io import BytesIO

import numpy as np
import requests
import tensorflow as tf

from flask import Flask, request, jsonify
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image

from tensorflow_scripts.apis import prediction_service_pb2

from tensorflow_scripts.utils import label_map_util
from tensorflow_scripts.utils.label_util import load_labels
from tensorflow_scripts.utils.img_util import classification_pre_process
from tensorflow_scripts.utils.img_util import object_detection_pre_process

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


@app.route('/vera_poles_trees/detect/', methods=['POST'])
def object_detection():
    # Obtém a imagem a partir do url path informado pelo cliente:
    # Converte o arquivo num float array
    response = request.json['data']
    formatted_json_input = object_detection_pre_process(response)

    # Create stub
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Create prediction request object
    request = predict_pb2.PredictRequest()

    # Specify model name (must be the same as when the TensorFlow serving serving was started)
    request.model_spec.name = 'obj_det'

    # Initalize prediction 
    # Specify signature name (should be the same as specified when exporting model)
    request.model_spec.signature_name = "detection_signature"
    request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto({FLAGS.input_image}))

    # Call the prediction server
    result = stub.Predict(request, 10.0)  # 10 secs timeout

    # Plot boxes on the input image
    category_index = label_map_util.load_label_map(FLAGS.path_to_labels)
    boxes = result.outputs['detection_boxes'].float_val
    classes = result.outputs['detection_classes'].float_val
    scores = result.outputs['detection_scores'].float_val
    # image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
    #     FLAGS.input_image,
    #     np.reshape(boxes,[100,4]),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8)

    # # Save inference to disk
    # scipy.misc.imsave('%s.jpg'%(FLAGS.input_image), image_vis)



    # TODO: Verificar se essa linha é necessária
    # this line is added because of a bug in tf_serving(1.10.0-dev)

    # Making POST request
    headers = {"content-type": "application/json"}
    response = requests.post(
        'http://localhost:8501/v1/models/vera_poles_trees:predict', headers=headers, data=formatted_json_input)

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
