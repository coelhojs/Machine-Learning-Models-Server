# Métodos utilitários utilizados na classificação de images e detecção de objetos
import json

import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow_scripts.utils.label_util import load_labels

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def classification_pre_process(image_path):
    """
    Pre-process the input image to return a json to pass to the tf model

    Args:
        image_path (str):  Path to the jpeg image

    Returns:
        formatted_json_input (str)
    """

    image = Image.open(image_path).convert("RGB")
    image_np = load_image_into_numpy_array(image)

    # Expand dims to create  bach of size 1
    image_tensor = np.expand_dims(image_np, 0)
    formatted_json_input = json.dumps(
        {"signature_name": "serving_default", "instances": image_tensor.tolist()})

    return formatted_json_input

def object_detection_pre_process(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = load_image_into_numpy_array(image)

    # Expand dims to create  bach of size 1
    image_tensor = np.expand_dims(image_np, 0)
    formatted_json_input = json.dumps({"signature_name": "serving_default", "instances": image_tensor.tolist()})

    return formatted_json_input

def post_process(server_response, image_size, labels_path):
    """
    Post-process the server response

    Args:
        server_response (requests.Response)
        image_size (tuple(int))

    Returns:
        post_processed_data (dict)
    """
    response = json.loads(server_response.text)
    output_dict = response['predictions'][0]


    # all outputs are float32 numpy arrays, so convert types as appropriate
    filtered_scores = list(filter(lambda x: (x > 0.5), output_dict['detection_scores']))
    output_dict['detection_scores'] = filtered_scores
    output_dict['num_detections'] = int(len(output_dict['detection_scores']))
    filtered_classes = output_dict['detection_classes'][0:output_dict['num_detections']]

    labels = load_labels(labels_path)

    named_classes = list(map(lambda x: labels[int(x)-1], filtered_classes))
    output_dict['detection_classes'] = named_classes
    filtered_boxes = output_dict['detection_boxes'][0:output_dict['num_detections']]
    output_dict['detection_boxes'] = filtered_boxes
    
    
    return output_dict
