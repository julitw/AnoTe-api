from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from models.database import get_db
from models.project_model import Project
import pandas as pd
from io import BytesIO
from fastapi.responses import StreamingResponse
import uuid
import json
from repositories.project_repository import add_project


async def add_new_project(
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
    modified_df = modified_df.rename(columns={column_text_name: 'text', column_label_name: 'label'})
    modified_df['predicted_label_by_llm'] = None
    modified_df['evaluated_label_by_user'] = None
    modified_df['was_annotated_by_user'] = None
    modified_df['was_annotated_by_user'] = modified_df['label'].notna().astype(int)
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
    add_project(db, new_project)
    
    
    return {"message": "Project is added!", "project_id": new_project.id}
    
    
    
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "id": project.id,
        "name": project.name,
        "column_text_name": project.column_text_name,
        "column_label_name": project.column_label_name,
        "available_labels": project.available_labels,
        "number_evaluated_data": project.number_evaluated_data ,
        "number_positive_evaluated_data": project.number_positive_evaluated_data ,
        "number_annotated_data": project.number_annotated_data,
        "total": project.row_count,
        "created_at": project.created_at,
    }
    
def downloadProject(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.file_data:
        raise HTTPException(status_code=404, detail="No file available in the database")

    file_like = BytesIO(project.modified_file_data)
    return StreamingResponse(file_like, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={project.file_name}"})


def get_data(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.modified_file_data:
        raise HTTPException(status_code=404, detail="No file available in the database")

    try:
        file_stream = BytesIO(project.modified_file_data)
        dataset_df = pd.read_csv(file_stream, encoding="utf-8")


        return dataset_df.astype(str).to_dict(orient='records')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")

def delete(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(project)
    db.commit()

    return {"message": "Project has been successfully deleted"}


async def get_file_columns(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file, nrows=1)
        columns = list(df.columns)
        print('column', columns)
        if not columns:
            return {"message": "The file does not contain headers"}
        return {"columns": columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading the file: {str(e)}")


async def get_labels(file: UploadFile = File(...), column_name: str = Form(...)):
    try:
        df = pd.read_csv(file.file)

        if column_name not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' does not exist in the file.")

        unique_values = df[column_name].dropna().unique().tolist()

        return {"unique_values": unique_values}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")