import numpy as np
import tensorflow as tf
from tensorflow_scripts.utils.label_util import load_labels

def load_graph(model_path):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_path, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(image_path,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(image_path, input_name)
  if image_path.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif image_path.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif image_path.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")

  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  result = sess.run(normalized)

  return result


def image_classifier(images_list, model_path, labels_path):
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"
  graph = load_graph(model_path)
  response = []
  for image_path in images_list:
    t = read_tensor_from_image_file(
        image_path,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.compat.v1.Session(graph=graph) as sess:
      inference = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })
    inference = np.squeeze(inference)
    results = []
    top_k = inference.argsort()[-5:][::-1]
    labels = load_labels(labels_path)
    for i in top_k:
        label = labels[i]
        score = float(inference[i])
        results.append('{label},{score}'.format(label=label,score=score))

    response.append(results)
    
  return response
