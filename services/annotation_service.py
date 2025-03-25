import json
import pandas as pd
from io import BytesIO
from fastapi import HTTPException
from sqlalchemy.orm import Session
from utils.annotation.prepare_data import prepare_dataframes
from utils.annotation.examples.ExamplesSelector import ExamplesSelector
from utils.annotation.llm.LLMAnnotator import LLMAnnotator
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

    examples_df = ExamplesSelector(
        dataset_df[dataset_df['was_annotated_by_user'] == 1]
    ).get_examples()

    annotator = LLMAnnotator(
        model=GPT4oMiniLLM(token=API_KEY, temperature=0),
        dataset=data_subset,
        examples_for_prompt=examples_df,
        prompt_template="Classify the text based on: {labels}. Return only label. Avoid explanations. \n\n{examples}\n\nText: {text}. ",
        text_column_name='text',
        labels=project.available_labels.split(',')
    )

    return generate_annotation_stream(annotator, db, project, dataset_df, data_subset)


def generate_annotation_stream(annotator, db, project, dataset_df, data_subset):
    for idx, result in zip(data_subset.index, annotator.get_results()):
        if isinstance(result, str):
            result = json.loads(result)

        dataset_df.at[idx, 'predicted_label_by_llm'] = result['predicted_label']
        dataset_df.at[idx, 'logprobs'] = json.dumps(result.get('logprobs', {}))
        dataset_df.at[idx, 'top_logprobs'] = json.dumps(result.get('top_logprobs', {}))

        save_dataframe_to_project(project, dataset_df, db)

        yield json.dumps({
            "id": str(dataset_df.at[idx, 'id']),
            "response": result['predicted_label'],
            "logprobs": result['logprobs'],
            "top_logprobs": result['top_logprobs']
        }) + "\n"


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
