import time
import requests

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from keras.applications.vgg16 import decode_predictions, preprocess_input
from tensorflow.keras.models import load_model
import io
import numpy as np
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from streamlit_lottie import st_lottie,st_lottie_spinner

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


load_anima= load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_4kmUDEKo63.json')



uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model = load_model('best_model.h5')
if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Create a button for analysis
    if st.button("Analyze"):
        with st_lottie_spinner(load_anima):

            # Preprocess the image
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG')
            image_bytes.seek(0)

            image = load_img(image_bytes, target_size=(224, 224))
            image = img_to_array(image)
            image = image / 255.0  # Rescale pixel values to [0, 1]

            # Add a batch dimension to the image
            image = tf.expand_dims(image, axis=0)

            # Make prediction
            predictions = model.predict(image)
            predicted_class = "Class 1" if predictions[0] > 0.5 else "Class 0"

            if predicted_class == "Class 0":
                agg_1 = 'jegan'
            elif predicted_class == "Class 1":
                agg_1 = 'richard'

            # Display the predicted class label
            st.write(f"Predicted Class: {agg_1}")