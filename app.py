from PIL import Image
import io
import numpy as np
import base64
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from io import BytesIO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


app = Flask(__name__)

# Load the model once when the app starts
model = load_model('MobileNetTyre.keras')

# Function to convert image to base64 string
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")  # Save the image to the buffer in JPEG format
    return base64.b64encode(buffered.getvalue()).decode("utf-8")  # Convert to base64 and return as string

@app.route('/')
def home():
    return render_template('index.html')  # This will render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    try:
        # Convert the file to a BytesIO object and then load the image
        img = Image.open(file)
        img = img.resize((224, 224))  # Resize to model's input size
        
        # Convert image to numpy array and preprocess for the model
        img_array = img_to_array(img) / 255.0  # Normalize if needed
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict the class
        prediction = model.predict(img_array)[0][0]  # Extract prediction score

        # Convert prediction to Python float (for JSON serialization)
        prediction = float(prediction)

        # Interpret the result
        threshold = 0.5  # Define a threshold for classification
        if prediction > threshold:
            result = {'class': 'good', 'confidence': prediction, 'image': image_to_base64(img)}
        else:
            result = {'class': 'defective', 'confidence': 1 - prediction, 'image': image_to_base64(img)}

        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    # Read the PORT environment variable (defaults to 8080)
    port = int(os.environ.get("PORT", 8080))
    
    # Ensure Flask listens on all IP addresses (0.0.0.0) and the correct port
    app.run(host="0.0.0.0", port=port)
