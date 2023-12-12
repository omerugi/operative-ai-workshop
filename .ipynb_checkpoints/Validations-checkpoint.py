import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
from io import BytesIO

def validate_dataset_loading(iris_dataset):
    try:
        assert iris_dataset is not None, "Dataset not loaded. Use the load_iris function."
        assert hasattr(iris_dataset, 'data') and hasattr(iris_dataset, 'target'), "Dataset seems incomplete."
        return "Dataset loaded correctly."
    except AssertionError as error:
        return error

def validate_type(train,test):
    try:
        if str(train.dtype) == 'float32' and str(test.dtype) == 'float32':
            return "Type changed succefully!"
        raise Exception(f"Wrong type: train type {str(train.dtype)} test type {str(test.dtype)}")
    except Exception as e: 
        return e


def validate_norm_rage(train_images,test_images):
    if train_images.min() == 0.0 and train_images.max() == 1.0 and test_images.min() == 0.0 and test_images.max() == 1.0:
        return "Normalization succeeded"
    raise Exception(f"Wrong values: train max/min {train_images.max()}{train_images.min()} test max/min {test_images.max()}{test_images.min()}")

def load_and_preprocess_image(url):
    response = requests.get(url)
    img = load_img(BytesIO(response.content), target_size=(32, 32))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array