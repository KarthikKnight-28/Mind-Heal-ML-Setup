import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

dataset_path = r"E:\karthck\Alzheimer_s Dataset\train"

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array
model = load_model("alzheimer_model.h5")

def predict_alzheimer(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    predicted_label = class_labels[np.argmax(predictions)]

    return predicted_label, predictions

input_image_path = r"K:\Dementia Cognizant\Alzheimer_s Dataset\train\MildDemented\mildDem4.jpg"
predicted_class, predictions = predict_alzheimer(input_image_path)

print("Predicted Class:", predicted_class)
print("Class Probabilities:", predictions)