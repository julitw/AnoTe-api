from io import BytesIO
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from models.database import get_db
from repositories.project_repository import get_project_or_404
from services.annotation_service import (
    stream_annotations,
    get_next_unannotated_ids,
    add_true_label_to_example,
    annotate_prompt_example_by_user
)
import pandas as pd
from utils.annotation.examples.ExamplesSelector import ExamplesSelector


router = APIRouter(
    prefix="/api/projects",
    tags=["Annotation"]
)


@router.post("/{project_id}/annotate")
def annotate_project(project_id: int, limit: int = 10, db: Session = Depends(get_db)):
    return StreamingResponse(
        stream_annotations(project_id, limit, db),
        media_type="application/plain"
    )


@router.get("/{project_id}/get-next-annotated-ids")
def get_next_annotated_ids(project_id: int, limit: int = 10, db: Session = Depends(get_db)):
    return get_next_unannotated_ids(project_id, limit, db)


@router.post("/{project_id}/add-true-label")
def add_true_label(project_id: int, exampleId: str, label: str, db: Session = Depends(get_db)):
    return add_true_label_to_example(project_id, exampleId, label, db)



@router.get("/{project_id}/high-entropy-llm-examples")
def get_high_entropy_llm_examples(project_id: int, top_k: int = 5, db: Session = Depends(get_db)):

    project = get_project_or_404(project_id, db)
    file_like = BytesIO(project.modified_file_data)
    df = pd.read_csv(file_like)

    selector = ExamplesSelector(df)
    top_examples = selector.get_high_entropy_llm_examples(top_k=top_k)

    # jeśli nie ma danych, zwróć pustą listę
    expected_cols = {'id', 'text', 'predicted_label_by_llm', 'top_logprobs'}
    if top_examples.empty or not expected_cols.issubset(set(top_examples.columns)):
        return []

    return top_examples[list(expected_cols)].astype(str).to_dict(orient='records')



@router.post("/{project_id}/annotate-prompt-example")
def annotate_prompt_example(project_id: int, example_id: str, true_label: str, db: Session = Depends(get_db)):
    return annotate_prompt_example_by_user(project_id, example_id, true_label, db)
