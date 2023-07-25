import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from streamlit_drawable_canvas import st_canvas


model = tf.keras.models.load_model('Model')

word_dict = {
    0:'A',
    1:'B',
    2:'C',
    3:'D',
    4:'E',
    5:'F',
    6:'G',
    7:'H',
    8:'I',
    9:'J',
    10:'K',
    11:'L',
    12:'M',
    13:'N',
    14:'O',
    15:'P',
    16:'Q',
    17:'R',
    18:'S',
    19:'T',
    20:'U',
    21:'V',
    22:'W',
    23:'X', 
    24:'Y',
    25:'Z'
}

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "point", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 30, 50, 30)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
bg_color = st.sidebar.color_picker("Background color hex: ", "#fff")

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color="#000",
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=400,
    width=400,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

def resize_image(image_array):
    image_pil = Image.fromarray(image_array)
    resized_pil = image_pil.resize((28, 28), Image.ANTIALIAS)
    grayscale_pil = resized_pil.convert("L")
    grayscale_array = np.array(grayscale_pil)
    _, img_thresh = cv2.threshold(grayscale_array, 100, 255, cv2.THRESH_BINARY_INV)

    return img_thresh

if st.button('Save Image'):
    plt.imshow(resize_image(canvas_result.image_data), cmap='gray')
    plt.savefig("saved_image.png")

if st.button('Predict'):
    img_final = resize_image(canvas_result.image_data)
    img_final = np.reshape(img_final, (1, 28, 28, 1))
    img_pred = word_dict[np.argmax(model.predict(img_final))]

    st.text(f'Prediction: {img_pred}')