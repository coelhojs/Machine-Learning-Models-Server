import json
import numpy as np

from utils import plot_util
from PIL import Image


def pre_process(image_path):
    """
    Pre-process the input image to return a json to pass to the tf model

    Args:
        image_path (str):  Path to the jpeg image

    Returns:
        formatted_json_input (str)
    """

    image = Image.open(image_path).convert("RGB")
    image_np = plot_util.load_image_into_numpy_array(image)

    # Expand dims to create  bach of size 1
    image_tensor = np.expand_dims(image_np, 0)
    image_tensor = np.expand_dims(image_np, 0)
    formatted_json_input = json.dumps(
        {"signature_name": "serving_default", "instances": image_tensor.tolist()})

    return formatted_json_input
