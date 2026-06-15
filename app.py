from flask import Flask, render_template, request, send_from_directory, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your model
try:
    model = load_model('model/davi3.0.keras')  # Ensure the model file is correctly located
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Map food classes to calorie values
calorie_map = {'cup_cakes': 200,'fried_rice': 350,'omelette': 150,'pizza': 285,'waffles': 220 }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/background.webp')
def background():
    return send_from_directory(os.getcwd(), 'background.webp')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load the image and preprocess it for prediction
    try:
        img = image.load_img(file_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])

        # Mapping class index to food item and calorie count
        class_labels = list(calorie_map.keys())
        predicted_class = class_labels[predicted_class_index]
        predicted_calories = calorie_map.get(predicted_class, "Calorie data not available")

        # Return the prediction and the path to the uploaded image
        return jsonify({
            'predicted_class': predicted_class,
            'calories': predicted_calories,
            'image_path': file.filename
        })

    except Exception as e:
        return jsonify({'error': f"Error processing image: {e}"}), 500

# Route to serve the uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False)  # Set debug to False for production environments
