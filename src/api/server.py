from fastapi import FastAPI

from src.api.inference_route import auth_router
from src.api.constants import APP_NAME, API_PREFIX

def server() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        docs_url=f"{API_PREFIX}/docs",
    )
    app.include_router(auth_router, prefix=API_PREFIX)
    return app