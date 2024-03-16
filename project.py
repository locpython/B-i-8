import numpy as np
import cv2
import streamlit as st
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# Set the title and instructions
st.title("Character Drawing Recognition App")
st.sidebar.write("Instructions")
st.sidebar.write("Please draw a character in the box below and press 'Predict'.")

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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # Chuyển từ RGBA sang RGB nếu cần
    resized_img = cv2.resize(img_rgb, (32, 32), interpolation=cv2.INTER_LANCZOS4)
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
    image = Image.open(uploaded_file)# cv2.cvtColor(uploaded_file, cv2.COLOR_RGBA2RGB)
    image_array = np.array(image)
    prediction = predict_img(image_array)
    resized_image = cv2.resize(image_array, (200, 200), interpolation=cv2.INTER_LANCZOS4)
    image_display = Image.fromarray(resized_image)
    col1.image(image_display, caption='Uploaded Image')# use_column_width=True)
    for line in prediction:
        col2.write(line)