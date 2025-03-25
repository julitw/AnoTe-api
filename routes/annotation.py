from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from models.database import get_db
from services.annotation_service import (
    stream_annotations,
    get_next_unannotated_ids,
    add_true_label_to_example
)

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
