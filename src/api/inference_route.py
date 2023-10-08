from fastapi import APIRouter

from src.api.schemas import Resume
from src.models.naive_bayes_model import NaiveBayesModel
from src.constants import NAIVE_BAYES_PIPELINE_PATH


model = NaiveBayesModel()
model.load(NAIVE_BAYES_PIPELINE_PATH)

inference_router = APIRouter()


@inference_router.post("/inference")
def run_inference(resume: Resume):
    prediction = model.predict([resume.text])
    return prediction.tolist()[0]