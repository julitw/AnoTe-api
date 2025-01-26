from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import project_routes

app = FastAPI()

# Zezwól na dostęp do API z frontendu Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(project_routes.router, prefix="/api/projects")
@app.get("/")
def root():
    return {"message": "API działa poprawnie"}
