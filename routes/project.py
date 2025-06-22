from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from models.database import get_db
from models.project_model import Project
import pandas as pd
from io import BytesIO
from fastapi.responses import StreamingResponse
import uuid
import json
from services.project_service import add_new_project, get_project, downloadProject, get_data, delete, get_file_columns, get_labels

router = APIRouter(
    prefix="/api/projects",
    tags=["Project"]
)


@router.post("/add")
async def add_project(
    name: str = Form(...),
    file: UploadFile = File(...),
    column_text_name: str = Form(...),
    column_label_name: str = Form(...),
    available_labels: str = Form(...),
    db: Session = Depends(get_db)
):
    
    response = await add_new_project(name, file, column_text_name, column_label_name, available_labels, db)
    return response
   

@router.get("/")
def get_projects(db: Session = Depends(get_db)):
    return db.query(Project).all()


@router.get("/{project_id}")
def get_project_by_id(project_id: int, db: Session = Depends(get_db)):
    return get_project(project_id, db) 


@router.get("/{project_id}/download")
def download_annotated_file(project_id: int, db: Session = Depends(get_db)):
    return downloadProject(project_id, db)


@router.get("/{project_id}/project-data")
def get_project_data(project_id: int, db: Session = Depends(get_db)):
    return get_data(project_id, db)

@router.delete("/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    return delete(project_id, db)


@router.post("/get_columns")
async def get_columns(file: UploadFile = File(...)):
    result = await get_file_columns(file)
    return result


@router.post("/get_unique_values")
async def get_unique_values(file: UploadFile = File(...), column_name: str = Form(...)):
    result = await get_labels(file, column_name)
    return result

@router.get("/{project_id}/label-map")
def get_label_map(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        label_map = json.loads(project.available_labels)
        return {"label_map": label_map}
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Label mapping is corrupted")
