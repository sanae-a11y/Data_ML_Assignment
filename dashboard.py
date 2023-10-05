import time
import streamlit as st


st.title("Resume Classification Dashboard")
st.sidebar.title("Dashboard Modes")

sidebar_options = st.sidebar.selectbox(
    "Options",
    ("EDA", "Training", "Inference")
)

if sidebar_options == "EDA":
    st.header("Exploratory Data Analysis")
    st.info("In this section, you are invited to create insightful graphs "
            "about the resume dataset that you were provided.")
elif sidebar_options == "Training":
    st.header("Model Training")
    st.info("Before you proceed to training your model. Make sure you "
            "have checked your training pipeline code and that it is set properly.")

    name = st.text_input('Model name', placeholder='Naive Bayes')
    serialize = st.checkbox('Save model')
    train = st.button('Train Model')

    if train:
        with st.spinner('Training model, please wait...'):
            time.sleep(1)
            st.write("Train executed")

else:
    st.header("Resume Inference")
    st.info("This section simplifies the inference process. "
            "Choose a test resume and observe the class that your model will predict."
    )
    
    infer = st.button('Run Resume Inference')

    if infer:
        with st.spinner('Running inference...'):
            time.sleep(1)
            st.write("Inference executed")
