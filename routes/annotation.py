
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models.database import get_db
from models.project_model import Project
import json
import pandas as pd
from io import BytesIO
from llm.LLMAnnotator import LLMAnnotator
from llm.models.GPT4omini import GPT4oMiniLLM

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("CLARIN_API_KEY")
router = APIRouter(
    prefix="/api/projects",
    tags=["Annotation"]
)


@router.post("/{project_id}/annotate")
def annotate_project(project_id: int, limit: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    start_index = project.last_annotated_index
    end_index = start_index + limit

    if not project.file_data:
        raise HTTPException(status_code=404, detail="No file available in the database")

    file_like = BytesIO(project.file_data)
    dataset_df = pd.read_csv(file_like)

    if start_index >= len(dataset_df):
        raise HTTPException(status_code=400, detail="All data has already been annotated.")

    data_subset = dataset_df.iloc[start_index:end_index]
    examples_df = data_subset[:5]


    annotator = LLMAnnotator(
        model=GPT4oMiniLLM(token=API_KEY, temperature=0),
        dataset=data_subset,
        examples_for_prompt=examples_df,
        prompt_template="Classify the text based on: {labels}. Return only label. Avoid explanations. \n\n{examples}\n\nText: {text}. ",
        text_column_name=project.column_text_name,
        labels=project.available_labels.split(',')
    )


    results = annotator.get_results()

    dataset_df.loc[start_index:end_index-1, 'predicted_label'] = [r['predicted_label'] for r in results]

    file_buffer = BytesIO()
    dataset_df.to_csv(file_buffer, index=False)
    project.file_data = file_buffer.getvalue()

    project.last_annotated_index = end_index
    db.commit()

    return {
        "message": "Annotation done",
        "results": results,
        "next_index": project.last_annotated_index
    }


@router.post("/{project_id}/add-true-label")
def add_true_label(project_id: int, index: int, label: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not project.file_data:
        raise HTTPException(status_code=404, detail="No file available in the database")
    
    try:
        file_stream = BytesIO(project.file_data)
        dataset_df = pd.read_csv(file_stream, encoding="utf-8")
        
        if 'true_label' not in dataset_df.columns:
            dataset_df['true_label'] = None  
        
        if index < 0 or index >= len(dataset_df):
            raise HTTPException(status_code=400, detail="Invalid index")
        
        dataset_df.at[index, 'true_label'] = label  
        
        file_buffer = BytesIO()
        dataset_df.to_csv(file_buffer, index=False)
        project.file_data = file_buffer.getvalue()
        db.commit()
        
        return {"message": "True label added successfully", "index": index, "label": label}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")