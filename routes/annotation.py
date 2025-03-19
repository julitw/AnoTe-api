
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from models.database import get_db
from models.project_model import Project

import pandas as pd
from io import BytesIO
import json
from dotenv import load_dotenv
import os

from utils.llm.LLMAnnotator import LLMAnnotator
from utils.llm.models.GPT4omini import GPT4oMiniLLM
from utils.annotation import prepare_dataframes, annotate_and_stream
from utils.examples.ExamplesSelector import ExamplesSelector


load_dotenv()

API_KEY = os.getenv("CLARIN_API_KEY")

router = APIRouter(
    prefix="/api/projects",
    tags=["Annotation"]
)


@router.post("/{project_id}/annotate")
def annotate_project(project_id: int, limit: int = 10, db: Session = Depends(get_db)):
    """
    Endpoint do anotacji projektu za pomocą modelu LLM.
    Strumieniuje wyniki i na bieżąco aktualizuje bazę danych.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.modified_file_data:
        raise HTTPException(status_code=404, detail="No file available in the database")

    file_like = BytesIO(project.modified_file_data)
    dataset_df = pd.read_csv(file_like)

    dataset_df = prepare_dataframes(dataset_df)

    unannotated_rows = dataset_df[(dataset_df['was_annotated_by_user'] == 0) & (dataset_df['predicted_label_by_llm'].isna())]
    if unannotated_rows.empty:
        raise HTTPException(status_code=400, detail="All data has already been annotated.")

    data_subset = unannotated_rows.iloc[:limit]
    if data_subset.empty:
        raise HTTPException(status_code=400, detail="No unannotated data available within the limit.")
    
    selector = ExamplesSelector(dataset_df[dataset_df['was_annotated_by_user'] == 1])
    examples_df = selector.get_examples()
    

    annotator = LLMAnnotator(
        model=GPT4oMiniLLM(token=API_KEY, temperature=0),
        dataset=data_subset,
        examples_for_prompt=examples_df,
        prompt_template="Classify the text based on: {labels}. Return only label. Avoid explanations. \n\n{examples}\n\nText: {text}. ",
        text_column_name='text',
        labels=project.available_labels.split(',')
    )

    return StreamingResponse(
        annotate_and_stream(annotator, db, project, dataset_df, data_subset),
        media_type="application/json"
    )


@router.get("/{project_id}/get-next-annotated-ids")
def get_next_annotated_ids(project_id: int, limit: int = 10, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not project.modified_file_data:
        raise HTTPException(status_code=404, detail="No file available in the database")
    
    file_like = BytesIO(project.modified_file_data)
    dataset_df = pd.read_csv(file_like)
    
    unannotated_rows = dataset_df[(dataset_df['was_annotated_by_user'] == 0) & (dataset_df['predicted_label_by_llm'].isna())]
    
    if unannotated_rows.empty:
        raise HTTPException(status_code=400, detail="All data has already been annotated.")
    
    data_subset = unannotated_rows.iloc[:limit]
    
    if data_subset.empty:
        raise HTTPException(status_code=400, detail="No unannotated data available within the limit.")
    
    updated_ids = dataset_df.loc[data_subset.index, 'id'].tolist()
    
    return {
        "message": "Success!",
        "updated_ids": updated_ids
    }
        
        
@router.post("/{project_id}/add-true-label")
def add_true_label(project_id: int, exampleId: str, label: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if not project.modified_file_data:
        raise HTTPException(status_code=404, detail="No file available in the database")
    
    try:
        file_stream = BytesIO(project.modified_file_data)
        dataset_df = pd.read_csv(file_stream, encoding="utf-8")
        
        if exampleId not in dataset_df['id'].astype(str).values:
            raise HTTPException(status_code=400, detail="Invalid ID")
        
        dataset_df.loc[dataset_df['id'].astype(str) == exampleId, 'evaluated_label_by_user'] = label  

        evaluated_data_number = int(dataset_df['evaluated_label_by_user'].notna().sum())
        positive_evaluated = int((dataset_df["predicted_label_by_llm"] == dataset_df["evaluated_label_by_user"]).sum())

        
        file_buffer = BytesIO()
        dataset_df.to_csv(file_buffer, index=False)
        project.modified_file_data = file_buffer.getvalue()
        project.number_positive_evaluated_data =positive_evaluated
        project.number_evaluated_data = evaluated_data_number
        db.commit()
        
        return {"message": "True label added successfully", "id": exampleId, "label": label}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")