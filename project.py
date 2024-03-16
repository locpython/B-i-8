import numpy as np
import os
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Set the title and instructions
st.title("Character Drawing Recognition App")
st.sidebar.write("Instructions")
st.sidebar.write("Please draw a character in the box below and press 'Predict'.")

# Configuration for the drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fill color changes for better visibility
    stroke_color="white",
    background_color="black",
    update_streamlit=True,
    width=200,
    height=200,
    drawing_mode="freedraw",
    stroke_width=20,  # Increased stroke width for clearer drawings
    key="canvas",
)

# Button to clear the canvas
if st.button('Clear Canvas'):
    st.experimental_rerun()

# Check if there is an image drawn
if canvas_result.image_data is not None:
    # Convert, resize, and normalize the image
    img_rgb = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    resized_img = cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_AREA)
    resized_img = resized_img / 255.0

    # Display the resized image dimensions
    st.write("Resized Image Shape:", resized_img.shape)

    # Button to predict the drawing
    if st.button('Predict'):
        # Load the model and make a prediction
        model = load_model("model.keras")
        prediction = model.predict(np.expand_dims(resized_img, axis=0))
        top_3_indices = np.argsort(prediction[0])[::-1][:3]

        letters = [chr(i) for i in range(65, 91)]
        st.write("Top 3 Predicted Characters:")
        for index in top_3_indices:
            predicted_letter = letters[index]
            confidence = prediction[0][index] * 100
            st.write(f"Character: {predicted_letter}, Confidence: {confidence:.2f}%")
