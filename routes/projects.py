from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from models.database import get_db
from models.project_model import Project
import json
import pandas as pd
from io import BytesIO
import uuid

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

    modified_df = dataset_df[[column_text_name, column_label_name]]
    modified_df = modified_df.rename(columns={column_text_name: 'text'})
    modified_df['predicted_label_by_llm'] = None
    modified_df['evaluated_label_by_user'] = None
    modified_df['was_annotated_by_user'] = None
    modified_df['was_annotated_by_user'] = modified_df[column_label_name].notna().astype(int)
    modified_df['id'] = [str(uuid.uuid4()) for _ in range(len(modified_df))]
    modified_file = BytesIO()
    modified_df.to_csv(modified_file, index=False, encoding='utf-8')
    modified_file.seek(0)

    new_project = Project(
        name=name,
        file_data=file_content,
        file_name=file.filename,
        column_text_name=column_text_name,
        column_label_name=column_label_name,
        available_labels=json.dumps(available_labels),
        row_count=len(dataset_df),
       modified_file_data = modified_file.getvalue(),
       number_annotated_data = int((modified_df["was_annotated_by_user"] == 1).sum())
       
       
    )

    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    
    return {"message": "Project is added!", "project_id": new_project.id}

@router.get("/")
def get_projects(db: Session = Depends(get_db)):
    return db.query(Project).all()
