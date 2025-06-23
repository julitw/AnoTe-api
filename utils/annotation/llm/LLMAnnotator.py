import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json

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
        
        self.prompt_with_examples = None
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
                    f"    Category:  {label_name}\n"
                )
                examples.append(example_str)

            result_string_examples = "\n\n".join(examples)
            

     
            prompt_template_with_examples = self.prompt_template.replace("{examples}", result_string_examples)
            
            
            if isinstance(self.labels, dict):
                label_lines = [f"{k} - {v}" for k, v in self.labels.items()]
                labels_string = "\n".join(label_lines)
            else:
                labels_string = ", ".join(self.labels)

            self.prompt_with_examples = prompt_template_with_examples.replace("{labels}", labels_string)


                        
            print('PROMPT', prompt_template_with_examples)


            self.prompt = PromptTemplate(
                input_variables=["text"],
                template=self.prompt_with_examples
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

    def fetch_answer(self, texts):
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
    
                # annotations.append({"text": text, 
                #                     "predicted_label": content , 
                #                     "logprobs": self.chain.llm.logprobs, 
                #                     "top_logprobs": self.chain.llm.logprobs['content'][0]['top_logprobs']})
                    yield {"text": text, 
                                    "predicted_label": content , 
                                    "logprobs": self.chain.llm.logprobs, 
                                    "top_logprobs": self.chain.llm.logprobs['content'][0]['top_logprobs']}

            except Exception as e:
                error_message = f"Error: {str(e)}"
                # annotations.append({"text": text, "annotation": error_message})
                yield {
                    "text": text,
                    "error": error_message
                }
        # return annotations


    def get_results(self):
        """
        Przeprowadza anotację dla tekstów w zbiorze danych.

        :return: Lista anotacji.
        """
        try:
            data = self.dataset_for_annotation
            if self.text_column_name not in data.columns:
                raise ValueError("Dataset must contain a 'text' column.")

            texts = data[self.text_column_name].tolist()
  

            self._build_prompt()
            self._build_chain()

            for result in self.fetch_answer(texts):
                yield json.dumps(result) + "\n"

        except Exception as e:
            yield json.dumps({"error", str(e)}) + "\n"
            
            
    def get_prompt(self):
        """
        Zwraca już zbudowany prompt.
        
        :return: Zbudowany prompt
        """
   
        return self.prompt_with_examples
        


