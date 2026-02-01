from flask import Flask, render_template, request, send_from_directory, url_for
from flask import url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2
import base64

app = Flask(__name__)

# Load the trained model
model_path = 'model_vgg16.h5'
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Flask routes
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/')
def new():
    return render_template('new8.html')

@app.route('/result')
def result():
    # Handle the result logic here
    return render_template('result.html')

# predict_image
def predict_image(imagepath, model, class_indices):
    try:
        # Load and preprocess image
        predict = image.load_img(imagepath, target_size = (224, 224))  
        predict_modified = image.img_to_array(predict)
        predict_modified = predict_modified / 255.0
        predict_modified = np.expand_dims(predict_modified, axis = 0)

        # Make prediction
        result = model.predict(predict_modified)
        
        # Get class index with highest probability
        predicted_class_index = np.argmax(result[0])
        probability = result[0][predicted_class_index]
        
        # Map index to class label using class_indices
        # Reverse the class_indices dictionary to get label from index
        labels = dict((v,k) for k,v in class_indices.items())
        prediction_class = labels[predicted_class_index]
        
        # Print results
        print("result", result)
        print("Prediction = " + prediction_class)
        print("Probability = {:.2f}%".format(probability * 100))
        
        return prediction_class, probability
    
    except Exception as e:
        print(f'Error predicting image: {e}')
        return None
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return "No file uploaded."
        
        # Save the uploaded file to the uploads directory
        filename = 'temp_image.jpg'
        file_path = os.path.join('uploads', filename)
        uploaded_file.save(file_path)
        print(f"Saved uploaded file to: {file_path}")

        class_indices={'diseased cotton leaf': 0,
                        'diseased cotton plant': 1,
                        'fresh cotton leaf': 2,
                        'fresh cotton plant': 3}
        
        prediction_class, probability = predict_image(file_path, model,class_indices)
        if prediction_class is None:
            return 'Prediction Failed'
        
        print(f"Predicted class: {prediction_class}")

        return render_template('result.html', prediction_class=prediction_class, prediction_probability=probability)
    except Exception as e:
        return f"Error: {e}"
    
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create the 'uploads' directory if it doesn't exist
    app.run(debug=True)