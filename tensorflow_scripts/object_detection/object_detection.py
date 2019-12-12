import json
import os
import pathlib
import sys

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from tensorflow_scripts.utils import img_util, label_map_util
from tensorflow_scripts.utils import ops as utils_ops

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# cd models/research/
# protoc object_detection/protos/*.proto --python_out=.


def load_model(model_path):
  graph = tf.Graph()
  with graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

  return graph

def objects_detector(images_list, model_path, labels):
    graph = load_model(model_path)
    response = []
    for image_path in images_list:
        inference = run_inference_for_single_image(graph, image_path, labels)

        response.append(inference)

    return response


"""Check the model's input signature, it expects a batch of 3-color images of type uint8:"""
#print(detection_model.inputs)

"""And retuns several outputs:"""

# detection_model.output_dtypes

# detection_model.output_shapes

"""Add a wrapper function to call the model, and cleanup the outputs:"""

def run_inference_for_single_image(graph, image_path, labels):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            try:
                image = Image.open(image_path).convert("RGB")
            except:
                raise Exception("Imagem {image_path} nao localizada.".format(image_path=image_path))

            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                            real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                            real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
            image_np = img_util.load_image_into_numpy_array(image)

            # Run inference
            output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image_np, 0)})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

            output_dict = img_util.post_process(output_dict, image_np.shape, labels)

    inference_dict = {}
    inference_dict['ImagePath'] = image_path
    inference_dict['Class'] = output_dict['detection_classes']
    inference_dict['BoundingBoxes'] = output_dict['detection_boxes'].tolist()
    inference_dict['Score'] = np.array(output_dict['detection_scores']).tolist()
    inference_dict['NumDetections'] = output_dict['num_detections']

    return inference_dict
