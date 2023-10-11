from pathlib import Path

PARENT_PATH = Path(__file__).parent.parent
print("ppppppppppppppppppppppp: ", PARENT_PATH)
DATA_PATH = PARENT_PATH / "data/"
RAW_DATASET_PATH = DATA_PATH / "raw/resume.csv"
PROCESSED_DATASET_PATH = DATA_PATH / "processed/resume.csv"

PROCESSED_DATA_PATH = PARENT_PATH / "processed/"

MODELS_PATH = PARENT_PATH / "models/"
NAIVE_BAYES_PIPELINE_PATH = MODELS_PATH / "naive_bayes_pipeline.joblib"
SVC_PIPELINE_PATH = MODELS_PATH / "svc_pipeline.joblib"
XGBC_PIPELINE_PATH = MODELS_PATH / "xgbc_pipeline.joblib"
REPORTS_PATH = PARENT_PATH / "reports/"
CM_PLOT_PATH = REPORTS_PATH / "cm_plot.png"
API_PID_PATH = PARENT_PATH / "api_pid.txt"
STREAMLIT_PID_PATH = PARENT_PATH / "streamlit_pid.txt"

SAMPLES_PATH = PARENT_PATH / "samples"
LABELS_MAP = {0: '.Net Developer', 1: 'Business Analyst', 2: 'Business Intelligence', 3: 'Help Desk and Support', 4: 'Informatica Developer', 5: 'Java Developer',
              6: 'Network and System Administrator', 7: 'Oracle DBA', 8: 'Project Manager', 9: 'Quality Assurance', 10: 'SAP', 11: 'SQL Developer', 12: 'Sharepoint Developer', 13: 'Web Developer'}
SAMPLES_MAP = {0: 'DOT_NET_DEVELOPER', 1: 'BUSINESS_ANALYST', 2: 'BUSINESS_INTELLIGENCE', 3: 'HELP_DESK_AND_SUPPORT', 4: 'INFORMATICA_DEVELOPER', 5: 'JAVA_DEVELOPER',
               6: 'NETWORK_AND_SYSTEM_ADMINISTRATOR', 7: 'ORACLE_DBA', 8: 'PROJECT_MANAGER', 9: 'QUALITY_ASSURANCE', 10: 'SAP', 11: 'SQL_DEVELOPER', 12: 'SHAREPOINT_DEVELOPER', 13: 'WEB_DEVELOPER'}
