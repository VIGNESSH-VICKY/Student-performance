import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("Student Performance Prediction")

# dataset create
np.random.seed(42)

study_hours = np.random.randint(1,10,1000)
sleep_hours = np.random.randint(4,9,1000)
attendance = np.random.randint(50,100,1000)

marks = study_hours*5 + sleep_hours*2 + attendance*0.5

data = pd.DataFrame({
    "StudyHours":study_hours,
    "SleepHours":sleep_hours,
    "Attendance":attendance,
    "Marks":marks
})

X = data[["StudyHours","SleepHours","Attendance"]]
y = data["Marks"]

model = LinearRegression()
model.fit(X,y)

st.header("Enter Student Details")

study = st.slider("Study Hours",0,10,5)
sleep = st.slider("Sleep Hours",0,10,6)
att = st.slider("Attendance",0,100,80)

if st.button("Predict Marks"):

    prediction = model.predict([[study,sleep,att]])

    st.success(f"Predicted Marks: {prediction[0]:.2f}")