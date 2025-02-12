import json
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
from fastapi.responses import StreamingResponse
from io import BytesIO

router = APIRouter()

Base.metadata.create_all(bind=engine)

# Funkcja do uzyskania sesji bazy danych
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



@router.post("/add")
async def add_project(
    name: str = Form(...),
    file: UploadFile = File(...),
    column_text_name: str = Form(...),
    column_label_name: str = Form(...),
    available_labels: str = Form(...),
    db: Session = Depends(get_db)  # 🔹 Pobranie sesji bazy danych
):
    print(f"DEBUG: name={name}, file={file.filename}, column_text_name={column_text_name}, column_label_name={column_label_name}, available_labels={available_labels}")

    # 🔹 Sprawdzenie, czy plik nie jest pusty
    file_content = await file.read()
    if not file_content.strip():
        raise HTTPException(status_code=400, detail="Przesłany plik jest pusty.")

    print("DEBUG: Otrzymano plik, pierwsze 500 bajtów:")
    print(file_content[:500].decode("utf-8", errors="replace"))  # Debugowanie zawartości pliku


    try:
        labels_list = json.loads(available_labels)
        if not isinstance(labels_list, list) or not all(isinstance(label, str) for label in labels_list):
            raise ValueError("Lista etykiet musi zawierać tylko stringi!")
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Niepoprawny format 'available_labels'.")

    # 🔹 Wczytanie CSV
    file_stream = BytesIO(file_content)

    try:
        dataset_df = pd.read_csv(file_stream, encoding="utf-8")
        if dataset_df.empty:
            raise HTTPException(status_code=400, detail="Plik CSV jest pusty.")

        print("DEBUG: Załadowano DataFrame:\n", dataset_df.head())

        # 🔹 Sprawdzenie, czy kolumny istnieją
        if column_text_name not in dataset_df.columns:
            raise HTTPException(status_code=400, detail=f"Kolumna '{column_text_name}' nie istnieje w pliku.")
        
        if column_label_name not in dataset_df.columns:
            raise HTTPException(status_code=400, detail=f"Kolumna '{column_label_name}' nie istnieje w pliku.")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nie udało się wczytać pliku CSV: {str(e)}")

    # 🔹 Dodanie projektu do bazy danych
    new_project = Project(
        name=name,
        file_data=file_content,  # Zapis pliku jako `LargeBinary`
        file_name=file.filename,  # Zapisujemy nazwę pliku
        column_text_name=column_text_name,
        column_label_name=column_label_name,
        available_labels=json.dumps(labels_list),  # Konwersja listy do stringa JSON
        last_annotated_index=0,  # Początkowy indeks anotacji
        row_count = len(dataset_df)
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

    if not project.file_data:
        raise HTTPException(status_code=404, detail="Brak pliku w bazie")

    # Wczytanie danych z binarnego pliku CSV
    file_like = BytesIO(project.file_data)
    dataset_df = pd.read_csv(file_like)

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
        prompt_template="Classify the text based on: {labels}. Return only label. Avoid explanations. \n\n{examples}\n\nText: {text}. ",
        text_column_name=project.column_text_name,
        labels=project.available_labels.split(',')
    )

    # Pobranie wyników anotacji
    results = annotator.get_results()

    # Dodanie wyników do danych
    dataset_df.loc[start_index:end_index-1, 'predicted_label'] = [r['predicted_label'] for r in results]

    # Zapisanie zmodyfikowanego pliku do bazy
    file_buffer = BytesIO()
    dataset_df.to_csv(file_buffer, index=False)
    project.file_data = file_buffer.getvalue()

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

    if not project.file_data:
        raise HTTPException(status_code=404, detail="Brak pliku w bazie")

    file_like = BytesIO(project.file_data)
    return StreamingResponse(file_like, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={project.file_name}"})




@router.get("/{project_id}/annotated-data")
def get_annotated_data(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projekt nie znaleziony")

    if not project.file_data:
        raise HTTPException(status_code=404, detail="Brak pliku w bazie danych")

    try:
        # 🔹 Odczyt pliku CSV z bazy
        file_stream = BytesIO(project.file_data)
        dataset_df = pd.read_csv(file_stream, encoding="utf-8")

        # 🔹 Jeśli `predicted_label` nie istnieje, zwróć pustą listę
        if 'predicted_label' not in dataset_df.columns:
            return []

        # 🔹 Pobranie tylko zaanotowanych danych
        annotated_data = dataset_df.iloc[:project.last_annotated_index]

        # 🔹 Jeśli nic nie było anotowane, zwróć pustą listę
        if annotated_data.empty:
            return []

        return annotated_data.astype(str).to_dict(orient='records')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas przetwarzania pliku: {str(e)}")

    

@router.delete("/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    
    if project is None:
        raise HTTPException(status_code=404, detail="Projekt nie został znaleziony")

    # Usunięcie projektu z bazy danych
    db.delete(project)
    db.commit()

    return {"message": "Projekt został pomyślnie usunięty"}


@router.post("/get_columns")
async def get_columns(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file, nrows=1)
        columns = list(df.columns)
        print('column', columns)
        if not columns:
            return {"message": "Plik nie zawiera  nagłówków"}
        return {"columns": columns}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas odczytu pliku: {str(e)}")


@router.post("/get_unique_values")
async def get_unique_values(file: UploadFile = File(...), column_name: str = Form(...)):
    try:

        df = pd.read_csv(file.file)

        if column_name not in df.columns:
            raise HTTPException(status_code=400, detail=f"Kolumna '{column_name}' nie istnieje w pliku.")

        unique_values = df[column_name].dropna().unique().tolist()

        return {"unique_values": unique_values}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas przetwarzania pliku: {str(e)}")
