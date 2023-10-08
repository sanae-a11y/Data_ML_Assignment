from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from src.models.base_model import BaseModel


class SVCModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            model=Pipeline([('svc', SVC(**kwargs))])
        )