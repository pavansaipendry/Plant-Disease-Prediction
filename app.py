
from PIL import Image
import tensorflow as tf  # or import torch if using PyTorch
import numpy as np
import json
import streamlit as st


model = tf.keras.models.load_model("/Users/pavansaipendry/Desktop/Master's/Sem 1/Deep Learning/Plant Disease Prediction with CNN/plane_disease_prediction_model.h5") 

class_indices = json.load(open(f"/Users/pavansaipendry/Desktop/Master's/Sem 1/Deep Learning/Plant Disease Prediction with CNN/class_indices.json"))

def new_image_conveter(image_path, target_size = (224,224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr , axis = 0)
    img_arr = img_arr.astype('float32') / 255.
    return img_arr

def predict_image_class(model, image_path, class_indices):
    preproc_img = new_image_conveter(image_path)
    predictions = model.predict(preproc_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App Interface
st.title("Plant Disease Prediction")
st.write("Upload a plant leaf image to predict the disease.")

# Image upload feature
uploaded_file = st.file_uploader("Choose an image...", type="JPG")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150,150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model , uploaded_file , class_indices)
            st.success(f'Prediction:{str(prediction)}')
