import numpy as np
import os
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Define the title and instructions of the app
st.title("Character Drawing Recognition App")
st.write("Please draw a character in the box below.")

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

if canvas_result.image_data is not None:
    # Convert the image from RGBA to RGB
    img_rgb = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2RGB)

    # Resize the image to match the model's expected input size
    resized_img = cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_AREA)

    # Normalize the image
    resized_img = resized_img / 255.0

    # Display the resized image dimensions
    st.write("Resized Image Shape:", resized_img.shape)

    # Load the model
    model = load_model("model.keras")

    # Make a prediction
    prediction = model.predict(np.expand_dims(resized_img, axis=0))
    top_3_indices = np.argsort(prediction[0])[::-1][:3]  # Get indices of top 3 probabilities in descending order

    letters = [chr(i) for i in range(65, 91)]  # Generate list of uppercase alphabet letters
    
    st.write("Top 3 Predicted Characters:")
    for index in top_3_indices:
        predicted_letter = letters[index]
        confidence = prediction[0][index] * 100
        st.write(f"Character: {predicted_letter}, Confidence: {confidence:.2f}%")
# link github.com
# https://github.com/locpython/Recognize-appp.git

# …or create a new repository on the command linef
# echo "# Recognize-appp" >> README.md
# git init
# git add README.md
# git commit -m "first commit"
# git branch -M mainff
# git remote add origin https://github.com/locpython/Recognize-appp.git
# git push -u origin main

# …or push an existing repository from the command line
# git remote add origin https://github.com/locpython/Recognize-appp.git
# git branch -M main
# git push -u origin main

# …or import code from another repository
# You can initialize this repository with code from a Subversion, Mercurial, or TFS project.

