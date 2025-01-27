from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from models.database import SessionLocal, engine
from models.project_model import Project, Base
import shutil
import os
from llm.LLMAnnotator import LLMAnnotator
from llm.models.GPT4omini import GPT4oMiniLLM
import pandas as pd
from fastapi.responses import FileResponse

router = APIRouter()

Base.metadata.create_all(bind=engine)

# Funkcja do uzyskania sesji bazy danych
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/add")
async def add_project(
    name: str = Form(...),
    file: UploadFile = File(...),
    column_text_name: str = Form(...),
    column_label_name: str = Form(...),
    available_labels: str = Form(...),
    db: Session = Depends(get_db)
):
    # Zapis pliku na serwerze
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        dataset_df = pd.read_csv(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Nie udało się wczytać pliku CSV.")

    # Sprawdzenie, czy kolumna istnieje w DataFrame
    if column_text_name not in dataset_df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Kolumna '{column_text_name}' nie istnieje w przesłanym pliku."
        )
    
    if column_label_name not in dataset_df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Kolumna '{column_label_name}' nie istnieje w przesłanym pliku."
        )

    # Zapis do bazy danych
    new_project = Project(
        name=name,
        file_path=file_path,
        column_text_name=column_text_name,
        column_label_name=column_label_name,
        available_labels=available_labels
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)

    return {"message": "Projekt został dodany!", "project_id": new_project.id}

@router.get("/")
def get_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).all()
    return projects


@router.get("/{project_id}")
def get_project_by_id(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Projekt nie znaleziony")
    return {
        "id": project.id,
        "name": project.name,
        "file_path": project.file_path,
        "column_text_name": project.column_text_name,
        "column_label_name": project.column_label_name,
        "available_labels": project.available_labels,
        "last_annotated_index": project.last_annotated_index
    }




@router.post("/{project_id}/annotate")
def annotate_project(project_id: int, limit: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Projekt nie znaleziony")

    start_index = project.last_annotated_index
    end_index = start_index + limit

    # Sprawdzenie czy plik istnieje
    if not os.path.exists(project.file_path):
        raise HTTPException(status_code=404, detail="Plik z danymi nie znaleziony")

    # Wczytanie danych
    dataset_df = pd.read_csv(project.file_path)

    if start_index >= len(dataset_df):
        raise HTTPException(status_code=400, detail="Wszystkie dane zostały już zaanotowane.")

    # Pobranie odpowiedniego zakresu danych
    data_subset = dataset_df.iloc[start_index:end_index]
    examples_df = data_subset[:5]

    # Uruchomienie modelu anotacji
    annotator = LLMAnnotator(
        model=GPT4oMiniLLM(token='IsrNlWB2APIENYsSA-tUW2WMlcM5WxSzstu_QQvYFzmWH0Q2', temperature=0),
        dataset=data_subset,
        examples_for_prompt=examples_df,
        prompt_template="Classify the text based on: {labels}. \n\n{examples}\n\nText: {text}",
        text_column_name=project.column_text_name,
        labels=project.available_labels.split(',')
    )

    # Pobranie wyników anotacji
    results = annotator.get_results()

    # Dodanie wyników do danych i zapisanie do pliku
    dataset_df.loc[start_index:end_index-1, 'predicted_label'] = [r['predicted_label'] for r in results]
    dataset_df.to_csv(project.file_path, index=False)

    # Zaktualizowanie ostatnio zaanotowanego indeksu
    project.last_annotated_index = end_index
    db.commit()

    return {
        "message": "Anotacja zakończona",
        "results": results,
        "next_index": project.last_annotated_index
    }

@router.get("/{project_id}/download")
def download_annotated_file(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Projekt nie znaleziony")

    # Sprawdzenie czy plik istnieje
    if not os.path.exists(project.file_path):
        raise HTTPException(status_code=404, detail="Plik z danymi nie znaleziony")

    return FileResponse(path=project.file_path, filename=f"annotated_project_{project_id}.csv", media_type="text/csv")



@router.get("/{project_id}/annotated-data")
def get_annotated_data(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projekt nie znaleziony")

    if not project.file_path:
        raise HTTPException(status_code=404, detail="Plik z danymi nie znaleziony")

    try:
        dataset_df = pd.read_csv(project.file_path)

        if 'predicted_label' not in dataset_df.columns:
            raise HTTPException(status_code=400, detail="Brak anotacji w pliku")

        # Pobranie danych do zaanotowanego indeksu
        annotated_data = dataset_df.iloc[:project.last_annotated_index]

        # Konwersja danych do formatów serializowalnych JSON
        annotated_data = annotated_data.astype(str).to_dict(orient='records')

        return annotated_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas przetwarzania pliku: {str(e)}")
    



@router.delete("/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    
    if project is None:
        raise HTTPException(status_code=404, detail="Projekt nie został znaleziony")

    # Usunięcie powiązanego pliku, jeśli istnieje
    if project.file_path and os.path.exists(project.file_path):
        os.remove(project.file_path)

    # Usunięcie projektu z bazy danych
    db.delete(project)
    db.commit()

    return {"message": "Projekt został pomyślnie usunięty"}