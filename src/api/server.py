from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware

from inference_route import inference_router
from api_constants import APP_NAME, API_PREFIX


def server() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        docs_url=f"{API_PREFIX}/docs",
    )
    app.include_router(inference_router, prefix=API_PREFIX)

    # Define allowed origins for CORS
    # Adjust this to your Streamlit app's URL
    origins = ["http://localhost:8000"]

    # Add CORS middleware to the FastAPI app
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],  # Adjust to the allowed HTTP methods
        allow_headers=["*"],  # Adjust to the allowed HTTP headers
    )

    return app
