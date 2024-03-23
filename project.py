import numpy as np
import cv2
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model

# Set the title
st.title("Character Drawing Recognition App")

# Create two columns with the specified ratios
col1, col2 = st.columns([2, 1])

# Configuration for the drawing canvas
# Use st.empty to create a placeholder to "reserve" the space for the canvas
with col1:
    canvas_placeholder = st.empty()

# Place file uploader in the second column
with col2:
    uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=['jpg', 'jpeg', 'png'], key='file_uploader')

# Now let's add the canvas to the placeholder in the first column
with canvas_placeholder.container():
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fill color changes for better visibility
        stroke_color="white",  # Stroke color for drawing
        background_color="black",  # Canvas background
        update_streamlit=True,
        width=200,
        height=200,
        drawing_mode="freedraw",
        stroke_width=20,
        key="canvas",
    )

def predict_img(img):
    img = np.array(img)
    # if img.shape[-1] == 4:  # Check if the image has alpha channel
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    resized_img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LANCZOS4)
    resized_img = resized_img / 255.0
    model = load_model("model.keras")
    prediction = model.predict(np.expand_dims(resized_img, axis=0))
    top_3_indices = np.argsort(prediction[0])[::-1][:3]
    letters = [chr(i) for i in range(65, 91)]
    predictions = []
    for index in top_3_indices:
        predicted_letter = letters[index]
        confidence = prediction[0][index] * 100
        predictions.append(f"Character: {predicted_letter}, Confidence: {confidence:.2f}%")
    return predictions    

if col1.button('Predict Drawing'):
    if canvas_result.image_data is not None:
        prediction = predict_img(canvas_result.image_data.astype(np.uint8))
        for line in prediction:
            st.write(line)

if col2.button('Predict Image') and uploaded_file is not None:
    # Open and convert the uploaded image to numpy array
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    # Predict the uploaded image
    prediction = predict_img(image_array)

    # Display prediction results
    for line in prediction:
        col2.write(line)
        
    # Display the uploaded image
    resized_image = cv2.resize(image_array, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    image_display = Image.fromarray(resized_image)
    col1.image(image_display, caption='Uploaded Image')

    # Display the grayscale image
    img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    img_gray_resized = cv2.resize(img_gray, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    img_gray_display = Image.fromarray(img_gray_resized, 'L')  # Specify mode as 'L' (grayscale)
    st.write(img_gray_display.size)
    col1.image(img_gray_display, caption='Gray Image')

    # Predict the uploaded grayscale image
    prediction_2 = predict_img(img_gray_display)
    
    # Add empty lines
    for _ in range(8):
        col2.write("")

    # Display the prediction results
    for line in prediction_2:
        col2.write(line)
