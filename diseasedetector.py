import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

model = load_model('/content/drive/MyDrive/ColabNotebooks/Healthcare/trainedmodel/my_model.h5')


class_labels = ['BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']


def preprocess_human_image(image_path):
   
    image = cv2.imread(image_path)
   
    image = cv2.resize(image, (224, 224))
    
    image = preprocess_input(np.expand_dims(image, axis=0))
    return image

def predict_skin_disease(image_path):
    preprocessed_image = preprocess_human_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    if predicted_class_index < len(class_labels):
        predicted_class_label = class_labels[predicted_class_index]
    else:
        predicted_class_label = "No match found"
    return predicted_class_label



