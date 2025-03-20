
from io import BytesIO
import json
import pandas as pd



def prepare_dataframes(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodaje brakujące kolumny do DataFrame, jeśli nie istnieją.
    """
    for column in ['predicted_label_by_llm', 'logprobs', 'top_logprobs']:
        if column not in dataset_df.columns:
            dataset_df[column] = None

    return dataset_df


def annotate_and_stream(annotator, db, project, dataset_df, data_subset):
    """
    Strumieniuje anotacje i na bieżąco zapisuje wyniki w bazie danych.
    """
    for idx, result in zip(data_subset.index, annotator.get_results()):
        update_database_with_results(db, project, dataset_df, idx, result)
        if isinstance(result, str):  # Jeśli `result` jest stringiem, dekoduj JSON
            result = json.loads(result)

        yield json.dumps({
            "id": str(dataset_df.at[idx, 'id']),
            "response": result['predicted_label'],
            "logprobs": result['logprobs'],
            "top_logprobs": result['top_logprobs']
        }) + "\n"


def update_database_with_results(db, project, dataset_df, idx, result):
    """
    Aktualizuje bazę danych po każdej anotacji.
    """
    try:
        if isinstance(result, str):  # Jeśli `result` jest stringiem, dekoduj JSON
            result = json.loads(result)
        dataset_df.at[idx, 'predicted_label_by_llm'] = result['predicted_label']
        dataset_df.at[idx, 'logprobs'] = json.dumps(result.get('logprobs', {}))
        dataset_df.at[idx, 'top_logprobs'] = json.dumps(result.get('top_logprobs', {}))

        file_buffer = BytesIO()
        dataset_df.to_csv(file_buffer, index=False)
        project.modified_file_data = file_buffer.getvalue()

        project.number_annotated_data = int(dataset_df['predicted_label_by_llm'].notna().sum()) + int((dataset_df["was_annotated_by_user"] == 1).sum())

        try:
            project = db.merge(project) 
            db.commit()
            db.refresh(project)
        except Exception as e:
            db.rollback()
            print(f"DB Commit Error: {e}")

    except Exception as e:
        db.rollback()
        print(f"Error updating database for record {idx}: {e}")