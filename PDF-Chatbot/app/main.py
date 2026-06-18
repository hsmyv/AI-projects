from fastapi import FastAPI
from app.routes.upload import router as upload_router
from app.routes.chat import router as chat_router
from app.core.config import APP_NAME
from fastapi.staticfiles import StaticFiles
from app.routes.web import router

app = FastAPI(title=APP_NAME)

app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

