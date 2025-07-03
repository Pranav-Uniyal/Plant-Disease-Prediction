import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Define class names / class indices
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load the trained model
model_path = r'modal\plant_disease_prediction_model.h5'  # add path of model according to your saved directory 
model = load_model(model_path)

# Ensure the upload directory exists
upload_dir = 'uploaded_images'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Function to make predictions
def predict_disease(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Streamlit App
st.title('Plant Disease Prediction 🌱🌼')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    try:
        # Display uploaded image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying.......")

        # Save the uploaded file
        img_path = os.path.join(upload_dir, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict disease
        predictions = predict_disease(img_path)
        if predictions is not None:
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index]
            st.write(f'Predicted  disease class: {predicted_class_name}')

    except Exception as e:
        st.error(f"An error occurred: {e}")

