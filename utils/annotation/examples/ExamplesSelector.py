import pandas as pd

class ExamplesSelector:
    def __init__(self, df: pd.DataFrame):
        """
        Inicjalizuje klasę z przekazanym DataFrame.
        """
        self.df = df

    def get_examples(self):
        """
        Zwraca pierwsze 10 rekordów z DataFrame.
        """
        return self.df[:10]
