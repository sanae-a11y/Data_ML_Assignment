from pathlib import Path

PARENT_PATH = Path(__file__).parent.parent

DATA_PATH = PARENT_PATH / "data/"
RAW_DATASET_PATH = DATA_PATH / "raw/resume.csv"
PROCESSED_DATA_PATH = PARENT_PATH / "processed/"

MODELS_PATH = PARENT_PATH / "models/"
REPORTS_PATH = PARENT_PATH / "reports/"