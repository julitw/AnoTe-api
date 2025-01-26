import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langsmith import traceable

class LLMAnnotator:
    def __init__(self, model, dataset, examples_for_prompt, prompt_template, text_column_name, labels):
        """
        Inicjalizuje klasę do interakcji z modelem LLM w celu anotacji danych.

        :param model: Obiekt modelu LLM.
        :param dataset:  dane do anotacji df.
        :param examples_for_prompt:  przykłady do prompta df.
        :param trace_name: Nazwa ścieżki do śledzenia procesu anotacji (może być pusty).
        :param prompt_template: treść prompta
        """
        self.model = model
        self.dataset_for_annotation = dataset
        self.examples_for_prompt = examples_for_prompt
        self.prompt_template = prompt_template
        self.text_column_name = text_column_name
        self.labels = labels

        self.prompt = None
        self.chain = None
        self.results = None

    def _build_prompt(self):
        """
        Tworzy prompt dla modelu LLM na podstawie podanego przykładu.
        """
        try:

            examples = []
            for idx, row in self.examples_for_prompt.iterrows():
                label_name = row['label']
                example_str = (
                    f"Example {idx + 1}:\n"
                    f"    Text: \"{row['text']}\"\n"
                    f"    Annotation:  {label_name}\n"
                )
                examples.append(example_str)

            result_string_examples = "\n\n".join(examples)

     
            prompt_template_with_examples = self.prompt_template.replace("{examples}", result_string_examples)
            prompt_template_with_examples = prompt_template_with_examples.replace("{labels}", ", ".join(self.labels))


            self.prompt = PromptTemplate(
                input_variables=["text"],
                template=prompt_template_with_examples
            )
        except Exception as e:
            raise ValueError(f"Error while building prompt: {e}")

    def _build_chain(self):
        """
        Tworzy łańcuch anotacji na podstawie modelu i prompta.
        """
        if not self.prompt:
            raise ValueError("Prompt has not been built. Call _build_prompt() first.")

        self.chain = LLMChain(llm=self.model, prompt=self.prompt)

    @traceable(name="traceable_annotation")
    def fetch_answer(self, texts, original_labels):
        """
        Przeprowadza anotację dla podanych tekstów.

        :param texts: Lista tekstów do anotacji.
        :return: Lista anotacji.
        """
        annotations = []

        for i, text in enumerate(texts):
            try:
                result = self.chain.run({"text": text})
                if isinstance(result, tuple):
                    content = result[0]  
                else:
                    content = result
                    print('Nr: ', i , "Predicted label: ", content)
                annotations.append({"text": text, 
                                    "predicted_label": content , 
                                    "logprobs": self.chain.llm.logprobs, 
                                    "top_logprobs": self.chain.llm.logprobs['content'][0]['top_logprobs'], 
                                    "original_label": original_labels[i]})


            except Exception as e:
                error_message = f"Error: {str(e)}"
                print('Nr: ', i, 'Error: ', error_message)
                annotations.append({"text": text, "annotation": error_message})
        return annotations


    def get_results(self):
        """
        Przeprowadza anotację dla tekstów w zbiorze danych.

        :return: Lista anotacji.
        """
        try:
            data = self.dataset_for_annotation
            if self.text_column_name not in data.columns:
                raise ValueError("Dataset must contain a 'text' column.")
            if 'label' not in data.columns:
                raise ValueError("Dataset must contain a 'label' column.")

            texts = data[self.text_column_name].tolist()
            labels = data['label'].tolist()

            self._build_prompt()
            self._build_chain()

            self.results = self.fetch_answer(texts, labels)
            return self.results
        except Exception as e:
            raise ValueError(f"Error in get_results: {e}")
        
    # def save_results(self, output_path):
    #     """
    #     Zapisuje wyniki anotacji do pliku CSV.

    #     :param output_path: Ścieżka do pliku, w którym zostaną zapisane wyniki.
    #     """
    #     if self.results is None:
    #         raise ValueError("No results available. Run get_results() first.")

    #     try:
    #         # Konwersja wyników do DataFrame
    #         results_df = pd.DataFrame(self.results)

    #         # Zapis do pliku CSV
    #         results_df.to_csv(output_path, index=False, encoding='utf-8')
    #         print(f"Results saved successfully to {output_path}")
    #     except Exception as e:
    #         raise ValueError(f"Error while saving results: {e}")


