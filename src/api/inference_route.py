from constants import NAIVE_BAYES_PIPELINE_PATH
from models.naive_bayes_model import NaiveBayesModel
from schemas import Resume
from fastapi import APIRouter
import sys
import os

# Get the path to the project's top-level directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project directory to sys.path
sys.path.insert(0, project_dir)


model = NaiveBayesModel()
model.load(NAIVE_BAYES_PIPELINE_PATH)

inference_router = APIRouter()


@ inference_router.post("/inference")
def run_inference(resume: Resume):
    prediction = model.predict([resume.text])
    return prediction.tolist()[0]
