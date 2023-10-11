from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from src.training.train_pipeline import TrainingPipeline
from src.constants import CM_PLOT_PATH, LABELS_MAP, SAMPLES_PATH, SAMPLES_MAP, RAW_DATASET_PATH, API_PID_PATH, STREAMLIT_PID_PATH
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

label_indices = {v: k for k, v in LABELS_MAP.items()}


def exploratory_data_analysis():
    st.header("Exploratory Data Analysis")
    # st.info("In this section, you can create insightful graphs about the resume dataset you were provided.")
    # Load and preprocess your dataset
    df = pd.read_csv(RAW_DATASET_PATH)

    # Display statistical descriptions
    st.subheader("Statistical Descriptions")
    st.write(df.describe())

    st.subheader("Statistical Descriptions by Resume Types")

    # Create a selectbox for the user to choose a label
    selected_label = st.selectbox("Select Type", list(label_indices.keys()))

    # Get the corresponding label index
    label_index = label_indices[selected_label]
    filtered_df = df[df['label'] == label_index]
    st.write(filtered_df.describe())

    # Create and display charts and visualizations
    st.subheader("Distribution of Resume Types")
    label_distribution = df['label'].map(LABELS_MAP).value_counts()
    # Define a color palette for the bars
    colors = sns.color_palette("Set3", len(label_distribution))

    fig, ax = plt.subplots()
    label_distribution.plot(kind='bar', ax=ax, color=colors)

    ax.set_xlabel('Resume Types')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    st.pyplot(fig)


def pipeline_training():
    st.header("Pipeline Training")
    model_type = st.selectbox('Select Model Type', [
                              'Naive Bayes', 'SVC', 'XGB Classifier'])
    name = st.text_input('Pipeline name', placeholder='Naive Bayes')
    serialize = st.checkbox('Save pipeline')
    train = st.button('Train pipeline')

    if train:
        with st.spinner('Training pipeline, please wait...'):
            try:
                model_mapping = {
                    'Naive Bayes': 'naive_bayes',
                    'SVC': 'svc',
                    'XGB Classifier': 'xgbc'
                }
                model_type = model_mapping.get(model_type)

                tp = TrainingPipeline(model_type)
                tp.train(serialize=serialize, model_name=name)
                tp.render_confusion_matrix()
                accuracy, f1 = tp.get_model_performance()
                col1, col2 = st.columns(2)
                if serialize:
                    st.success(f"Pipeline saved !")
                else:
                    st.warning(
                        "Don't forget to save the pipeline if needed. You can repeat the training anytime 🙂.")
                st.write("Model Type: ", model_type)
                col1.metric(label="Accuracy score",
                            value=str(round(accuracy, 4)))
                col2.metric(label="F1 score", value=str(round(f1, 4)))

                st.image(Image.open(CM_PLOT_PATH), width=850)
            except Exception as e:
                st.error('Failed to train the pipeline!')
                st.exception(e)


# Create an SQLite database
engine = create_engine('sqlite:///predictions.db')
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    label = Column(String)
    timestamp = Column(DateTime)


def save_prediction_to_db(label):
    prediction = Prediction(label=label, timestamp=datetime.now())
    session.add(prediction)
    session.commit()


def resume_inference():
    st.header("Resume Inference")
    sample = st.selectbox(
        "Resume samples for inference",
        tuple(LABELS_MAP.values()),
        index=None,
        placeholder="Select a resume sample",
    )
    infer = st.button('Run Inference')

    if infer:
        with st.spinner('Running inference...'):
            try:
                def get_key_by_label(label, label_map):
                    for key, value in label_map.items():
                        if value == label:
                            return key
                    return None
                sample_file = SAMPLES_MAP[get_key_by_label(
                    sample, LABELS_MAP)] + ".txt"

                with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                    sample_text = file.read()

                try:
                    with open(API_PID_PATH, "r") as pid_file:
                        fastapi_pid = int(pid_file.readline().strip())
                    print("API PID:", fastapi_pid)
                except FileNotFoundError:
                    print("api_pid.txt file not found.")
                except ValueError:
                    print("Failed to read a valid PID from api_pid.txt.")

                try:
                    with open(STREAMLIT_PID_PATH, "r") as pid_file:
                        streamlit_pid = int(pid_file.readline().strip())
                    print("API PID:", streamlit_pid)
                except FileNotFoundError:
                    print("api_pid.txt file not found.")
                except ValueError:
                    print("Failed to read a valid PID from  streamlit_pid.txt.")

                params = {"streamlit_pid": streamlit_pid,
                          "fastapi_pid": fastapi_pid}
                result = requests.post(
                    f'http://localhost:8000/api/inference?pid={fastapi_pid}',
                    json={'text': sample_text},

                )
                st.write("PIIIIIIIIIIIIIIDDDDDDDDDDDD", fastapi_pid)
                st.success('Done!')
                st.write("The Resume", sample_text)
                label = LABELS_MAP.get(int(float(result.text)))
                st.metric(label="Status", value=f"Resume label: {label}")
            except Exception as e:
                st.error('Failed to call Inference API!')
                st.exception(e)


# Main Streamlit app title
st.title("Resume Classification Dashboard")
st.sidebar.title("Dashboard Modes")

# Sidebar options and section selection
sidebar_options = st.sidebar.selectbox(
    "Options",
    ("EDA", "Training", "Inference")
)

# Dispatch to the appropriate section based on user choice
if sidebar_options == "EDA":
    exploratory_data_analysis()
elif sidebar_options == "Training":
    pipeline_training()
else:
    resume_inference()
