from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('models/model.h5')

# Class labels in the exact order mapped during training
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Define the uploads folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to predict tumor type
def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    # Apply MobileNetV2 preprocessing (scale to [-1, 1])
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    # Map the index to classification label
    predicted_label = class_labels[predicted_class_index]
    
    if predicted_label == 'notumor':
        return "No Tumor", confidence_score
    else:
        # e.g., "Tumor: glioma"
        return f"Tumor: {predicted_label}", confidence_score

# Route for the main page (index.html)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file:
            # Save the file
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            # Predict the tumor
            result, confidence = predict_tumor(file_location)

            # Return result along with image path for display
            # Note: We pass confidence as a numeric string (e.g. 98.54) without '%', as expected by the UI.
            return render_template(
                'index.html', 
                result=result, 
                confidence=f"{confidence * 100:.2f}", 
                file_path=f'/uploads/{file.filename}'
            )

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to serve sample files from the root directory
@app.route('/samples/<filename>')
def get_sample_file(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)

