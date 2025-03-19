from fastapi import FastAPI
from models.database import engine, Base
from routes import  project, annotation
from fastapi.middleware.cors import CORSMiddleware
from routes import project, annotation

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(project.router)
app.include_router(annotation.router)


@app.get("/")
def root():
    return {"message": "API works!"}
