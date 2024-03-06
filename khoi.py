import streamlit as st
import pandas as pd
import numpy as np
 sklearn.linear_model
st.title("Revenue Prediction")
st.balloons()
st.snow()
pred = st.slider('Input Temperature', 0, 45, 25)
df = pd.read_csv("C:\\Users\\lusan\\OneDrive\\Desktop\\IceCreamData.csv")
x = df['Temperature'].values.reshape(-1, 1)
y = df['Revenue'].values.reshape(-1, 1)
model = sklearn.linear_model.LinearRegression()
model.fit(x, y)
y_pred = model.predict([[pred]])
k = y_pred[0][0]
st.write(f"Revenue Prediction: {round(k,1)}")

