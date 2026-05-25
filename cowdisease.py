import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


cow_model = load_model('/content/drive/MyDrive/ColabNotebooks/Healthcare/trainedmodel/skin_cancer_detection_model.keras')
cow_threshold = 0.5


def preprocess_cow_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255. 
    return img_array






def classify_cow_disease(image_path):

  preprocessed_image= preprocess_cow_image(image_path)
  predictions = cow_model.predict(preprocessed_image)
  if predictions[0][0] <= cow_threshold: 
    return "Lumpy Skin"
  elif predictions[0][0] == 0:
    return None
  else:
    return "Normal"

