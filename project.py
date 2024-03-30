import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten, Input
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
import bz2

def load_and_preprocess_data():
    with bz2.open('dataset_X.pkl.bz2', 'rb') as file:
        X = pickle.load(file)
    with bz2.open('dataset_y.pkl.bz2', 'rb') as file:
        y = pickle.load(file)
    return X, y

def build_and_train_model(X_train, y_train, num_classes, epochs=10, test_size=0.5):
    # Preprocess labels
    label_encoder = LabelEncoder()
    y_train_encoded = to_categorical(label_encoder.fit_transform(y_train))
    
    # Define model
    model = Sequential([
        Input(shape=X_train.shape[1:]),
        Flatten(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Split the data
    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X_train, y_train_encoded, test_size=test_size, random_state=42
    )
    
    # Train the model
    history = model.fit(X_train, y_train_encoded, epochs=epochs, validation_split=0.1, verbose=1)
    
    # Evaluate the model
    evaluation_result = model.evaluate(X_test, y_test_encoded)
    
    return model, history, evaluation_result

def display_training_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(history.history['loss'], label='Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epochs')
    
    axs[1].plot(history.history['accuracy'], label='Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epochs')
    
    st.pyplot(fig)

def show_dataset_examples(X, y):
    # Assuming y is already processed to single digit per label if necessary
    classes = np.unique(y)
    fig, axs = plt.subplots(len(classes), 8, figsize=(10, 2 * len(classes)))
    for i, class_label in enumerate(classes):
        indices = np.where(y == class_label)[0]
        for j in range(8):
            if len(indices) > j:
                img = X[indices[j]]
                axs[i, j].imshow(img, cmap='gray')
                axs[i, j].axis('off')
    st.pyplot(fig)
    
def draw_test_confident (model):  
    canvas_placeholder = st.empty()
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
    if canvas_result.image_data is not None:
        prediction = predict_img(canvas_result.image_data.astype(np.uint8), model)
        for line in prediction:
            st.write(line)
            
def predict_img(img, model):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    resized_img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LANCZOS4)
    resized_img = resized_img / 255.0
    prediction = model.predict(np.expand_dims(resized_img, axis=0))
    top_3_indices = np.argsort(prediction[0])[::-1][:3]
    letters = [chr(i) for i in range(65, 91)]
    predictions = []
    for index in top_3_indices:
        predicted_letter = letters[index]
        confidence = prediction[0][index] * 100
        predictions.append(f"Character: {predicted_letter}, Confidence: {confidence:.2f}%")
    return predictions 
   
def main():
    st.title("Ứng dụng Streamlit với nhiều trang")
    menu = ["Trang Chính", "Xem Dataset", ]
    choice = st.sidebar.selectbox("Chọn trang", menu)

    X, y = load_and_preprocess_data()

    if choice == "Trang Chính":
        st.header("Huấn luyện mô hình nhận dạng ký tự")
        epochs = st.number_input('Chọn số epochs:', min_value=1, max_value=100, value=10)
        test_size = st.slider('Tỉ lệ dữ liệu kiểm tra:', min_value=0.1, max_value=0.9, value=0.2)
        
        if st.button("Huấn luyện mô hình"):
            model, history, eval_result = build_and_train_model(X, y, num_classes=np.unique(y).shape[0], epochs=epochs, test_size=test_size)
            st.write(f"Độ chính xác trên tập kiểm tra: {eval_result[1]:.2f}")
            display_training_history(history)
            draw_test_confident(model)
    
    elif choice == "Xem Dataset":
        st.header("Hiển thị mẫu từ Dataset")
        show_dataset_examples(X, y)

if __name__ == "__main__":
    main()
