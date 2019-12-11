import tensorflow as tf

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label