from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from pyngrok import ngrok
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="/content/drive/MyDrive/ColabNotebooks/Healthcare/templates/")
 # Load the trained model
model = load_model('trainedmodel/skin_cancer_detection_model.keras')
# Define the threshold value
threshold = 0.5
# Function to preprocess user-inputted image
def preprocess_image(image_path, target_size=(150, 150)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize the pixel values
    return img_array
# Function to classify the image
def classify_image(predictions, threshold):
    if predictions <= threshold:
        return "Disease"  # Classify as disease if predicted probability is above or equal to threshold
    else:
        return "Normal"  # Classify as normal otherwise
@app.route('/')
def index():
    return render_template('index.html')
# Route to handle image upload and classification
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    # Save the uploaded file to a temporary directory
    filename = secure_filename(file.filename)
    filepath = f'/tmp/{filename}'
    file.save(filepath)
    # Preprocess the uploaded image
    input_image = preprocess_image(filepath)
    # Perform inference
    predictions = model.predict(input_image)
    # Classify the image
    classification = classify_image(predictions, threshold)
    return render_template('result.html', classification=classification)
    



def start_ngrok():
    public_url = ngrok.connect(5000)
    print(' * Tunnel URL:', public_url)

if __name__ == '__main__':
    start_ngrok()
    app.run()