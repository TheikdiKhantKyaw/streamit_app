import streamlit as st
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import io

# Define constants
TARGET_SIZE = (224, 224)
CLASS_LABELS = {0: 'Clams',
 1: '🦀 Crabs',
 2: '🐬 Dolphin',
 3: '🐟 Eel',
 4: '🐟 Fish',
 5: '🪼 Jelly Fish',
 6: '🦞 Lobster',
 7: '🐙 Octopus',
 8: '🦦 Otter',
 9: '🐧 Penguin',
 10: '🐡 Puffers',
 11: '🐠 Sea Rays',
 12: '✴ Sea Urchins',
 13: '🌊🐴 Seahorse',
 14: '🦭 Seal',
 15: '🦈 Sharks',
 16: '🦐 Shrimp',
 17: '🦑 Squid',
 18: '⭐ Starfish',
 19: '🐢 Turtle_Tortoise',
 20: '🐳 Whale'}

# Load the pre-trained model
model_path = 'my_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
except ValueError as e:
    st.error(f"Error loading model: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to preprocess image
def preprocess_image(image):
    img = tf.image.resize(image, TARGET_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to predict image class
def predict_image(model, image, class_labels):
    # Preprocess the image
    img_array = preprocess_image(image)

    # Make prediction
    predictions = model.predict(img_array)

    # Get predicted class index
    predicted_class = np.argmax(predictions)

    # Get predicted class label
    predicted_label = class_labels.get(predicted_class, "Unknown")

    return  predicted_label

# Create Streamlit app
st.title("Sea Life Image Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    image = np.array(image)

    predicted_label = predict_image(model, image, CLASS_LABELS)

    st.text(f"It is {predicted_label}")