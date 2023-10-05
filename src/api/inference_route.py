from fastapi import APIRouter

from src.api.schemas import Resume


inference_router = APIRouter()


@inference_router.get("/")
def index():
    return "Inference index route"


@inference_router.post("/inference")
def run_inference(resume: Resume):
    return "Inference value"