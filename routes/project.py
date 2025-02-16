from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from models.database import get_db
from models.project_model import Project
import pandas as pd
from io import BytesIO
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix="/api/projects",
    tags=["Project"]
)

@router.get("/{project_id}")
def get_project_by_id(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "id": project.id,
        "name": project.name,
        "column_text_name": project.column_text_name,
        "column_label_name": project.column_label_name,
        "available_labels": project.available_labels,
        "last_annotated_index": project.last_annotated_index
    }


@router.get("/{project_id}/download")
def download_annotated_file(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.file_data:
        raise HTTPException(status_code=404, detail="No file available in the database")

    file_like = BytesIO(project.file_data)
    return StreamingResponse(file_like, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={project.file_name}"})


@router.get("/{project_id}/annotated-data")
def get_annotated_data(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.file_data:
        raise HTTPException(status_code=404, detail="No file available in the database")

    try:
        file_stream = BytesIO(project.file_data)
        dataset_df = pd.read_csv(file_stream, encoding="utf-8")

        if 'predicted_label' not in dataset_df.columns:
            return []

        annotated_data = dataset_df.iloc[:project.last_annotated_index]

        if annotated_data.empty:
            return []

        return annotated_data.astype(str).to_dict(orient='records')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")


@router.delete("/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    
    db.delete(project)
    db.commit()

    return {"message": "Project has been successfully deleted"}


@router.post("/get_columns")
async def get_columns(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file, nrows=1)
        columns = list(df.columns)
        print('column', columns)
        if not columns:
            return {"message": "The file does not contain headers"}
        return {"columns": columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading the file: {str(e)}")


@router.post("/get_unique_values")
async def get_unique_values(file: UploadFile = File(...), column_name: str = Form(...)):
    try:
        df = pd.read_csv(file.file)

        if column_name not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column_name}' does not exist in the file.")

        unique_values = df[column_name].dropna().unique().tolist()

        return {"unique_values": unique_values}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")
