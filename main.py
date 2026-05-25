from flask import Flask, render_template, render_template_string, request, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from pyngrok import ngrok
from werkzeug.utils import secure_filename
import cv2
from diseasedetector import predict_skin_disease, preprocess_human_image
from cowdisease import preprocess_cow_image, classify_cow_disease
from coviddetector import predict_covid_disease, preprocess_covid_image
import os


app = Flask(__name__, static_folder="/content/drive/MyDrive/ColabNotebooks/Healthcare/templates/")

def process_image(image_path):
  cow_classification = classify_cow_disease(image_path)
  human_classification = predict_skin_disease(image_path)
  covid_classification = predict_covid_disease(image_path)
  
 
  return cow_classification, human_classification, covid_classification

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse')
def analyze():
  return render_template('imageprocess.html')



@app.route('/humanbody', methods=['POST'])
def humanbody():

  data = request.json
  selected_body_parts = data.get("bodyParts") if data else None
  print(selected_body_parts)

  return redirect(url_for('result_page'))


@app.route('/process_image_route', methods=['POST'])
def process_image_route():


  if 'image' not in request.files:
    return jsonify({'error': 'No file uploaded'})
  file = request.files['image']
  if file.filename == '':
    return jsonify({'error': 'No file selected'})
  filename = secure_filename(file.filename)
  filepath = f'/tmp/{filename}'
  global resultfile
  resultfile = f'/tmp/{filename}'
  file.save(filepath)
  save_path = '/content/drive/MyDrive/fftemp'
  save_filepath = os.path.join(save_path, filename)
  try:
    file.save(save_filepath)
    print(f"Image saved to Google Drive: {save_filepath}")
  except Exception as e:
    print(f"Error saving image to Google Drive: {e}")
  
  global cow_classification, human_classification, covid_classification
  
  cow_classification, human_classification, covid_classification = process_image(filepath)
  return redirect(url_for('select_body'))
  

@app.route('/selectbody')
def select_body() :

  return render_template('humanbody.html')


@app.route('/result')
def result_page():
  return render_template('result.html', cow_classification=cow_classification, human_classification=human_classification, covid_classification=covid_classification)

def start_ngrok():
    public_url = ngrok.connect(5000)
    print(' * Tunnel URL:', public_url)

if __name__ == '__main__':
    start_ngrok()
    app.run()
