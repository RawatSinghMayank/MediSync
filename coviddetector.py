import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

model= load_model('/content/drive/MyDrive/ColabNotebooks/Healthcare/trainedmodel/covidmodel.h5')
class_names=['Normal','Covid','Pneumonia']


def preprocess_covid_image(image_path):
  image=cv2.imread(image_path)
  image=cv2.resize(image,( 50,50))
  image = np.expand_dims(image, axis=0)
  image = image / 255.0  
  return image

def predict_covid_disease(image_path):
  preprocessed_image= preprocess_covid_image(image_path)
  predictions=model.predict(preprocessed_image)
  prediction_class_index = np.argmax(predictions)
  predicted_class = class_names[prediction_class_index]

  return predicted_class



