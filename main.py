from fastapi import FastAPI
from models.database import engine, Base
from routes import projects, project, annotation
from fastapi.middleware.cors import CORSMiddleware
from routes import projects, project, annotation

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(project.router)
app.include_router(annotation.router)


@app.get("/")
def root():
    return {"message": "API works!"}
