
import pandas as pd
import json
from repositories.project_repository import get_project_or_404
from fastapi import HTTPException
from sqlalchemy.orm import Session
from io import BytesIO
from utils.explanation.explanation import calculate_explanation


def get_explanation(project_id: int, example_id: str, db: Session):
    # Wyszukiwanie projektu
    project = get_project_or_404(project_id, db)

    # Wczytanie pliku CSV z bazy danych
    file_like = BytesIO(project.modified_file_data)
    df = pd.read_csv(file_like)

    # Wyszukiwanie wiersza po ID
    example_row = df[df['id'] == example_id]

    if example_row.empty:
        raise HTTPException(status_code=404, detail="Example not found")

    # Zbieranie danych z projektu
    text = example_row['text'].iloc[0]
    logprobs = example_row['logprobs'].iloc[0]
    top_logprobs = example_row['top_logprobs'].iloc[0]
    used_prompt = example_row['used_prompt'].iloc[0]  # Dodanie odczytu kolumny 'used_prompt'

    # Sprawdzenie, czy logprobs są zapisane w formacie JSON, jeśli tak - deserializujemy
    try:
        logprobs = json.loads(logprobs) if isinstance(logprobs, str) else logprobs
        top_logprobs = json.loads(top_logprobs) if isinstance(top_logprobs, str) else top_logprobs
    except json.JSONDecodeError:
        logprobs = {}
        top_logprobs = {}

    # Pobranie dostępnych etykiet z projektu
    available_labels = json.loads(project.available_labels) if isinstance(project.available_labels, str) else project.available_labels

    # Wyświetlanie logprobs, tekstu, użytego promptu oraz etykiet
    print('TEXT:', text)  # Wyświetlenie tekstu
    print('LOGPROBS:', top_logprobs)  # Wyświetlenie logprobs
    print('USED PROMPT:', used_prompt)  # Wyświetlenie użytego promptu
    print('AVAILABLE LABELS:', available_labels)  # Wyświetlenie dostępnych etykiet

    text = calculate_explanation(text, top_logprobs, used_prompt, available_labels)
    
    return text