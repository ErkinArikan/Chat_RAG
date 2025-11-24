from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.routes import router as api_router

app = FastAPI(title="RAG Doküman Asistanı API")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(api_router)