from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from models.database import get_db
from models.project_model import Project
import json
import pandas as pd
from io import BytesIO

router = APIRouter(
    prefix="/api/projects",
    tags=["Projects"]
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
    file_content = await file.read()
    if not file_content.strip():
        raise HTTPException(status_code=400, detail="File is empty.")

    dataset_df = pd.read_csv(BytesIO(file_content), encoding="utf-8")


    if 'predicted_label' not in dataset_df.columns:
        dataset_df['predicted_label']= None
    if 'true_label' not in dataset_df.columns:
            dataset_df['true_label'] = 'None'

    new_project = Project(
        name=name,
        file_data=file_content,
        file_name=file.filename,
        column_text_name=column_text_name,
        column_label_name=column_label_name,
        available_labels=json.dumps(available_labels),
        last_annotated_index=0,
        row_count=len(dataset_df)
    )

    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    
    return {"message": "Project is added!", "project_id": new_project.id}

@router.get("/")
def get_projects(db: Session = Depends(get_db)):
    return db.query(Project).all()
