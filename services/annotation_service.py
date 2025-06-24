import json
import pandas as pd
from io import BytesIO
from fastapi import HTTPException
from sqlalchemy.orm import Session
from utils.annotation.prepare_data import prepare_dataframes
from utils.annotation.examples.ExamplesSelector import ExamplesSelector
from utils.annotation.llm.LLMAnnotator import LLMAnnotator
from utils.annotation.examples.ExamplesSelector import ExamplesSelector
from utils.annotation.llm.models.GPT4omini import GPT4oMiniLLM
from repositories.project_repository import get_project_or_404, save_dataframe_to_project
import os

API_KEY = os.getenv("CLARIN_API_KEY")

def stream_annotations(project_id: int, limit: int, db: Session):
    project = get_project_or_404(project_id, db)

    file_like = BytesIO(project.modified_file_data)
    dataset_df = pd.read_csv(file_like)
    dataset_df = prepare_dataframes(dataset_df)

    unannotated_rows = dataset_df[
        (dataset_df['was_annotated_by_user'] == 0) &
        (dataset_df['predicted_label_by_llm'].isna())
    ]

    if unannotated_rows.empty:
        raise HTTPException(status_code=400, detail="All data has already been annotated.")

    data_subset = unannotated_rows.iloc[:limit]
    if data_subset.empty:
        raise HTTPException(status_code=400, detail="No unannotated data available within the limit.")

    examples_df = create_examples(dataset_df)

    annotator = LLMAnnotator(
        model=GPT4oMiniLLM(token=API_KEY, temperature=0),
        dataset=data_subset,
        examples_for_prompt=examples_df,
        prompt_template="Classify the text based on: \n\n {labels}.  \n\n{examples}\n\nText: {text}. \n\n Return only one label  (like 0,1,2). Avoid explanations. ",
        text_column_name='text',
        labels = {int(k): v for k, v in json.loads(project.available_labels).items()}

    )
    
    print('OK', annotator.get_prompt())

    return generate_annotation_stream(annotator, db, project, dataset_df, data_subset)



def generate_annotation_stream(annotator, db, project, dataset_df, data_subset):
    for idx, result in zip(data_subset.index, annotator.get_results()):
        try:
            if isinstance(result, str):
                result = json.loads(result)
                
                
            print(result)

            # Sprawdź, czy klucz 'predicted_label' istnieje
            if 'predicted_label' not in result:
                raise KeyError("Missing 'predicted_label' in result")

            # Aktualizacja DataFrame
            dataset_df.at[idx, 'predicted_label_by_llm'] = result['predicted_label']
            dataset_df.at[idx, 'logprobs'] = json.dumps(result.get('logprobs', {}))
            dataset_df.at[idx, 'top_logprobs'] = json.dumps(result.get('top_logprobs', {}))
            dataset_df.at[idx, 'used_prompt'] = annotator.get_prompt()

            # Zapisz zmiany w projekcie
            save_dataframe_to_project(project, dataset_df, db)
        
            
            print(idx, result['predicted_label'], flush=True)
            print('')
            print(idx, result['top_logprobs'], flush=True)

            # Zwróć wynik jako strumień
            yield json.dumps({
                "id": str(dataset_df.at[idx, 'id']),
                "prompt": annotator.get_prompt(),
                "response": result['predicted_label'],
                "logprobs": result.get('logprobs', {}),
                "top_logprobs": result.get('top_logprobs', {})
            }) + "\n"

        except KeyError as e:
            # Obsługa brakujących kluczy w wyniku
            print(f"KeyError: {e}, skipping index {idx}", flush=True)
            continue
        except json.JSONDecodeError as e:
            # Obsługa błędów parsowania JSON
            print(f"JSONDecodeError: {e}, skipping index {idx}", flush=True)
            continue
        except Exception as e:
            # Obsługa innych nieoczekiwanych błędów
            print(f"Unexpected error: {e}, skipping index {idx}", flush=True)
            continue


def get_next_unannotated_ids(project_id: int, limit: int, db: Session):
    project = get_project_or_404(project_id, db)

    file_like = BytesIO(project.modified_file_data)
    df = pd.read_csv(file_like)

    unannotated = df[
        (df['was_annotated_by_user'] == 0) &
        (df['predicted_label_by_llm'].isna())
    ]

    if unannotated.empty:
        raise HTTPException(status_code=400, detail="All data has already been annotated.")

    subset = unannotated.iloc[:limit]
    return {
        "message": "Success!",
        "updated_ids": df.loc[subset.index, 'id'].tolist()
    }


def add_true_label_to_example(project_id: int, example_id: str, label: str, db: Session):
    project = get_project_or_404(project_id, db)

    file_like = BytesIO(project.modified_file_data)
    df = pd.read_csv(file_like)

    if example_id not in df['id'].astype(str).values:
        raise HTTPException(status_code=400, detail="Invalid ID")

    df.loc[df['id'].astype(str) == example_id, 'evaluated_label_by_user'] = label

    evaluated_count = int(df['evaluated_label_by_user'].notna().sum())
    positive_count = int((df["predicted_label_by_llm"] == df["evaluated_label_by_user"]).sum())

    project.number_evaluated_data = evaluated_count
    project.number_positive_evaluated_data = positive_count

    save_dataframe_to_project(project, df, db)

    return {"message": "True label added successfully", "id": example_id, "label": label}



def annotate_prompt_example_by_user(project_id: int, example_id: str, true_label: str, db: Session):
    project = get_project_or_404(project_id, db)
    file_like = BytesIO(project.modified_file_data)
    df = pd.read_csv(file_like)

    if example_id not in df['id'].astype(str).values:
        raise HTTPException(status_code=404, detail="Example not found")

    df.loc[df['id'].astype(str) == example_id, 'evaluated_label_by_user'] = true_label
    df.loc[df['id'].astype(str) == example_id, 'selected_as_prompt_example'] = 1

    save_dataframe_to_project(project, df, db)

    return {"message": "Example annotated and marked as prompt reference", "id": example_id}



def create_examples(dataset_df: pd.DataFrame) -> pd.DataFrame:
    # Przykłady oznaczone ręcznie
    was_annotated = dataset_df[
        dataset_df['was_annotated_by_user'] == 1
    ][['text', 'label']].copy()
    was_annotated['evaluated_label_by_user'] = None  # brak tej kolumny w tym zestawie

    # Przykłady wybrane jako prompt example (ręcznie ocenione)
    prompt_examples = dataset_df[
        dataset_df['selected_as_prompt_example'] == 1
    ][['text', 'evaluated_label_by_user']].copy()
    prompt_examples = prompt_examples.rename(columns={'evaluated_label_by_user': 'label'})
    prompt_examples['evaluated_label_by_user'] = prompt_examples['label']  # kopia do dodatkowej kolumny
    

    # Połącz oba
    examples_df = pd.concat([was_annotated, prompt_examples], ignore_index=True)
    
        
    for col in ['label']:
        examples_df[col] = examples_df[col].apply(lambda x: str(int(x)) if pd.notna(x) else "")
    
    return examples_df




def get_logprobs_for_text(project_id: int, example_id: str, db: Session):
    # Wyszukiwanie projektu
    project = get_project_or_404(project_id, db)

    # Wczytanie pliku CSV
    file_like = BytesIO(project.modified_file_data)
    df = pd.read_csv(file_like)

    # Wyszukiwanie wiersza po ID
    example_row = df[df['id'] == example_id]

    if example_row.empty:
        raise HTTPException(status_code=404, detail="Example not found")

    # Zbieranie danych z projektu
    text = example_row['text'].iloc[0]

    # Tworzymy annotatora LLM
    examples_df = create_examples(df)
    annotator = LLMAnnotator(
        model=GPT4oMiniLLM(token=API_KEY, temperature=0),
        dataset=pd.DataFrame(),  # Nie używamy danych do anotacji, bo klasyfikujemy jeden wiersz
        examples_for_prompt=examples_df,
        prompt_template="Classify the text based on: {labels}. Return only label. Avoid explanations. \n\n{examples}\n\nText: {text}. ",
        text_column_name='text',
        labels=project.available_labels.split(',')
    )

    # Otrzymanie wyników
    result = annotator.annotate_single_text(text)

    return result




