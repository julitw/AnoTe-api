import pandas as pd
import numpy as np
import json

class ExamplesSelector:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_high_entropy_llm_examples(self, top_k: int = 5):
        df = self.df.copy()

        # Jeśli nie ma potrzebnych kolumn, zwróć pustą ramkę
        if not all(col in df.columns for col in ['predicted_label_by_llm', 'top_logprobs', 'evaluated_label_by_user']):
            return pd.DataFrame()

        df = df[
            df['predicted_label_by_llm'].notna() &
            df['top_logprobs'].notna() &
            df['evaluated_label_by_user'].isna()
        ]

        if df.empty:
            return pd.DataFrame()

        df['entropy'] = df['top_logprobs'].apply(compute_entropy)

        return df.sort_values(by='entropy', ascending=False).head(top_k)




        


def compute_entropy(top_logprobs_json: str) -> float:
    try:
        parsed = json.loads(top_logprobs_json)

        label_logprobs = {}

        if isinstance(parsed, list):
            for entry in parsed:
                token = entry.get("token", "").strip()
                logprob = entry.get("logprob", None)

                if token and isinstance(logprob, (int, float)):
                    label_logprobs[token] = float(logprob)

        if not label_logprobs:
            raise ValueError("No usable logprobs")

        probs = np.exp(list(label_logprobs.values()))
        probs /= np.sum(probs)

        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return entropy

    except Exception as e:
        print(f"[Entropy Error] {e}")
        return -1
