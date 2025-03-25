from fastapi import HTTPException
from sqlalchemy.orm import Session
from io import BytesIO
import pandas as pd
from models.project_model import Project

def get_project_or_404(project_id: int, db: Session) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.modified_file_data:
        raise HTTPException(status_code=404, detail="No file available in the database")
    return project

def save_dataframe_to_project(project: Project, df: pd.DataFrame, db: Session):
    try:
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        project.modified_file_data = buffer.getvalue()

        project.number_annotated_data = int(df['predicted_label_by_llm'].notna().sum()) + \
                                        int((df["was_annotated_by_user"] == 1).sum())

        if not db.object_session(project):
            project = db.merge(project)

        db.add(project)
        db.commit()
        db.refresh(project)
    except Exception as e:
        db.rollback()
        print(f"Error saving project data: {e}")


def add_project(db:Session, new_project: Project):
        db.add(new_project)
        db.commit()
        db.refresh(new_project)