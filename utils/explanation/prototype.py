import pandas as pd
from openai import OpenAI
import json
import math
import re
import pandas as pd
import numpy as np
import random
import html
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

class PromptBuilder:
    def __init__(self, prompt: str, classes: list):
        self.prompt = prompt
        self.classes = classes
        
    def format_examples(self, examples):
        formatted = "### Examples:\n"
        for i, ex in enumerate(examples, 1):
            formatted += f"{i}. Text: {ex['text']}\n   {ex['class_index']}\n\n"
        return formatted.strip()
    
    def create_prompt(self, text, examples=None):

        classes_str = ''
        for i, cls in enumerate(self.classes):
            classes_str += f"{i + 1}. {cls}\n"
        classes_str += "\n"
        
        num_values_list = [f"'{i+1}'" for i in range(len(self.classes))]
        if len(num_values_list) > 1:
            num_values = ', '.join(num_values_list[:-1]) + f" or {num_values_list[-1]}"
        else:
            num_values = num_values_list[0]  
            
        if examples is None:
            prompt = self.prompt.format(classes=classes_str, text=text, num_values=num_values, examples='')
        else:
            text_example = self.format_examples(examples)
            prompt = self.prompt.format(classes=classes_str, text=text, num_values=num_values, examples=text_example)

        return prompt


class OpenAIWrapper:
    def __init__(self,  model: str, api_key: str = None, base_url: str = None, temperature: float = 0.0,
                logit_bias: dict = None, max_tokens: int = 1000, logprobs: bool = True,
                top_logprobs: int = 5, classes: list = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.logit_bias = logit_bias if logit_bias else {}
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.classes = classes if classes else []
        self.result = None
        self.client = None
    
    def connect(self):
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        if self.client.api_key is None:
            raise ValueError("API key is not set. Please provide a valid API key.")
        if self.client.base_url is None:
            raise ValueError("Base URL is not set. Please provide a valid base URL.")
        else:
            print(f"Connected to OpenAI API at {self.client.base_url}")


    def generate_response(self, prompt: str, system_context: str):
        if self.client is None:
            self.connect()
            if self.client is None:
                return {'error': True, 'message': 'Failed to connect to OpenAI'}

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_context},
                        {"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                logit_bias=self.logit_bias,
                logprobs=self.logprobs,
                top_logprobs=self.top_logprobs,
            )
            return response
        except Exception as e:
            print(f"Error generating response from OpenAI: {e}")
            return e
    
    def extract_probabilities(self, top_probabilities):
        probabilities = {}
        known_probabilities = []
        
        for item in top_probabilities:
            token = item.token
            prob = math.exp(item.logprob)
            probabilities[token] = prob
            known_probabilities.append(prob)
        
        estimated_probabilities = {}
        for class_name in range(1, len(self.classes) + 1):
            class_name = str(class_name)
            if class_name in probabilities:
                estimated_probabilities[class_name] = probabilities[class_name]
            else:
                known_probabilities = list(probabilities.values())
                if known_probabilities:
                    estimated_probabilities[class_name] = min(known_probabilities)
                else:
                    estimated_probabilities[class_name] = 0.0  
        
        total_probability = sum(probabilities.values())
        for token in probabilities:
            probabilities[token] /= total_probability if total_probability > 0 else 1
        
        return estimated_probabilities
    
    def parse_response(self, response, **kwargs):
        parsed = {}

        if isinstance(response, dict) and response.get('error'):
            parsed['content'] = -1
            parsed['json'] = {}
            parsed.setdefault('logprobs', [])
            parsed.setdefault('probability', [])
            print(f"Error encountered during response generation: {response['message']}")
            return parsed

        if kwargs.get('content', True):
            try:
                parsed['content'] = response.choices[0].message.content
                if kwargs.get('json', False):
                    match = re.search(r"```json\n(.+?)\n```", parsed['content'], re.DOTALL)
                    if match:
                        json_text = match.group(1)
                    else:
                        json_text = parsed['content']
                    try:
                        json_text = html.unescape(json_text)
                        json_text = json_text.replace("“", "\"").replace("”", "\"").replace("’", "'")

                        parsed['json'] = json.loads(json_text)
                    except json.JSONDecodeError:
                        print("Failed to parse JSON from response content.")
                        parsed['json'] = {}
            except AttributeError:
                print("Error: 'NoneType' object has no attribute 'choices' in parse_response (likely due to a previous error).")
                parsed['content'] = -1
                parsed['json'] = {}

        if kwargs.get('logprobs', False):
            try:
                if response.choices[0].logprobs and response.choices[0].logprobs.content and response.choices[0].logprobs.content[0] and response.choices[0].logprobs.content[0].top_logprobs:
                    parsed['logprobs'] = response.choices[0].logprobs.content[0].top_logprobs
                else:
                    parsed['logprobs'] = []
            except AttributeError:
                parsed['logprobs'] = []

            if kwargs.get('probability', False):
                parsed['probability'] = self.extract_probabilities(parsed.get('logprobs', []))

        if kwargs.get('self_explanation', False):
            try:
                parsed['self_explanation'] = response.choices[0].message.content
            except AttributeError:
                parsed['self_explanation'] = ""
        
        self.result = parsed
        return parsed
        
    
    def save_response(self, file_path: str, record: dict):
        with open(file_path, 'a') as file:
            json.dump(record, file, indent=4)
            file.write('\n')
        print(f"Response saved to {file_path}")
        
                