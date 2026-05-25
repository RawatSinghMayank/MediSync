from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from pyngrok import ngrok
from werkzeug.utils import secure_filename
import cv2
from diseasedetector import predict_skin_disease, preprocess_human_image
from cowdisease import preprocess_cow_image, classify_cow_disease


app = Flask(__name__, static_folder="/content/drive/MyDrive/ColabNotebooks/Healthcare/templates/")

def process_image(image_path):
  cow_classification = classify_cow_disease(image_path)
  human_classification = predict_skin_disease(image_path)
  
  return cow_classification, human_classification



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image_route', methods=['POST'])
def process_image_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    filename = secure_filename(file.filename)
    filepath = f'/tmp/{filename}'
    file.save(filepath)

   
    
    cow_classification, human_classification = process_image(filepath)

    if cow_classification == "Lumpy Skin":
        human_classification = None
    elif human_classification is not  None:
        cow_classification = None
    

    return render_template('result.html', cow_classification=cow_classification, human_classification=human_classification)

def start_ngrok():
    public_url = ngrok.connect(5000)
    print(' * Tunnel URL:', public_url)

if __name__ == '__main__':
    start_ngrok()
    app.run()
