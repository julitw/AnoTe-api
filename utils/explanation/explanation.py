import pandas as pd
import math
import shap
import numpy as np
import time
from openai import OpenAI
# from prototype import PromptBuilder, OpenAIWrapper
from .prototype import OpenAIWrapper
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("CLARIN_API_KEY")
BASE_URL = 'https://services.clarin-pl.eu/api/v1/oapi'

def get_logprobs(top_logrprobs, all_labels):
    probabilities = {}
    known_probabilities = []
            
    for item in top_logrprobs:
        token = item['token']
        prob = math.exp(item['logprob'])
        probabilities[token] = prob
        known_probabilities.append(prob)
        
    classes_names = list(all_labels.keys())
    
    estimated_probabilities = {}
    for class_name in range(0, len(classes_names)):
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


def llm_classifier_probabilities(texts, prompt_base, all_labels, logit_bias, api_key, base_url):    
    openai_wrapper = OpenAIWrapper(
        model='gpt-4o-mini',
        temperature=0.0,
        max_tokens=1,
        top_logprobs=10,
        classes=[i for i in range(0, len(all_labels.keys()))],
        logit_bias=logit_bias,
        api_key=api_key,
        base_url=base_url       
    )
    
    probabilities = []
    for text  in texts:
        prompt = prompt_base.format(text=text)
        
        response = openai_wrapper.generate_response(prompt=prompt, system_context='You are a helpful assistant that classifies text into specific categories based on the provided examples.')
        
        response_details = openai_wrapper.parse_response(
                response=response,
                logprobs=True,
                probability=True)
        
        probabilities.append(
            [prob for prob in response_details['probability'].values()]
        )
        time.sleep(0.01)
    
    probabilities = np.array(probabilities)
    
    return probabilities    

def get_SHAP_values(text, prompt, logit_bias, all_labels, api_key, base_url):
    
    masker = shap.maskers.Text(
        mask_token='[MASK]'
    )
    llm_prob = lambda texts: llm_classifier_probabilities(texts, prompt, all_labels, logit_bias, api_key, base_url)
    
    explainer = shap.Explainer(
        llm_prob,
        masker,
        output_names=list(all_labels.values())
    )
    
    shap_values = explainer([text])
    
    return shap_values

def generate_prompts_with_masking(text, prompt_base):
    words = text.split()
    prompts = []
    for i in range(len(words)):
        masked_text = ' '.join(words[:i] + ['[MASK]'] + words[i+1:])
        prompts.append(prompt_base.format(text=masked_text))
    return prompts, words

def compute_importance_scores(records, top_logrprobs, original_class):
    if isinstance(top_logrprobs, dict):
        f_c_x = top_logrprobs.get(original_class, 0.0)
    
    results = []
    for entry in records:
        f_c_masked = entry.get("probability", {}).get(original_class, 0.0)
        importance = f_c_x - f_c_masked
        results.append({
            "token": entry.get("mask", ""),
            "predicted_class": original_class,
            "importance": importance
        })

    return pd.DataFrame(results)


def get_occlusion_values(text, prompt_base, logit_bias, all_labels, api_key, base_url, top_logrprobs, original_class):
    
    results_occlusion = []
    prompts, words = generate_prompts_with_masking(text, prompt_base)
    
    openaiwrapper = OpenAIWrapper(
        model='gpt-4o-mini',
        temperature=0.0,
        max_tokens=1,
        top_logprobs=10,
        classes=[i for i in range(0, len(all_labels.keys()))],
        logit_bias=logit_bias,
        api_key=api_key,
        base_url=base_url
    )
    
    for i, prompt in enumerate(prompts):
        response = openaiwrapper.generate_response(prompt=prompt, system_context='You are a helpful assistant that classifies text into specific categories based on the provided examples.')
        
        response_details = openaiwrapper.parse_response(
            response=response,
            logprobs=True,
            probability=True,
            content=True
        )
        
        record = {
            'text': text,
            'mask': words[i],
            'probability': response_details['probability'],
            'predicted_class': response_details['content']
        }
        results_occlusion.append(record)
        time.sleep(0.01)
        
    results_importance = compute_importance_scores(results_occlusion, top_logrprobs, original_class)
    return results_importance


def get_self_explanations(text, prompt_etp, all_labels, api_key, base_url):

    openai_wrapper = OpenAIWrapper(
        model='gpt-4o-mini',
        temperature=0.0,
        max_tokens=1000,
        top_logprobs=10,    
        classes=[i for i in range(0, len(all_labels.keys()))],
        api_key=api_key,
        base_url=base_url)
    
    prompt = prompt_etp.format(text=text)
    
    response = openai_wrapper.generate_response(prompt=prompt, system_context='You are a helpful assistant that classifies text and explains your reasoning.')
    
    response_details = openai_wrapper.parse_response(
        response=response,
        json=True,
        content=True
    )
    
    record = {
        'text': text,
        'predicted_class': response_details['json']['category'],
        'prediction_probability': response_details['json']['confidence'],
        'words_probability': response_details['json']['word_importance']
    }
    
    return record


def get_top_k_words_combined(
    occlusion_df,
    # shap_words,
    sv,
    self1_words_probs,
    # self2_words_probs,
    k_ratio=0.2,
    shap_weight=2
):
    tokens = occlusion_df.sort_values('token').token.tolist()
    n_words = len(tokens)
    k = max(1, int(np.ceil(k_ratio * n_words)))

    words_1, probs_1 = zip(*self1_words_probs)
    words_1 = [w.lower() for w in words_1]
    probs_1 = np.array(probs_1, dtype=float)

    gt_class = np.argmax(sv.values[0].sum(axis=0))
    shap_words = sv.data[0]
    shap_values = sv.values[0][:, gt_class]
    
    all_words = set(tokens) | set(shap_words) | set(words_1) 

    def get_ranking(words, importances):
        order = np.argsort(-np.abs(importances))
        return {words[i]: rank for rank, i in enumerate(order)}

    rank_okluzja = get_ranking(tokens, occlusion_df['importance'].values)
    rank_shap = get_ranking(shap_words, shap_values)
    rank_self1 = get_ranking(words_1, probs_1)
    # rank_self2 = get_ranking(words_2, probs_2)

    combined = []
    for word in all_words:
        r1 = rank_okluzja.get(word, n_words)
        r2 = rank_shap.get(word, n_words)
        r3 = rank_self1.get(word, n_words)
        # r4 = rank_self2.get(word, n_words)
        total = r1 + shap_weight * r2 + r3 # + r4
        combined.append((word, total))
    combined.sort(key=lambda x: x[1])

    top_k = [w for w, _ in combined[:k]]

    return top_k


def get_final_explanation(top_k_words, record, api_key, base_url):
    prompt_base = """
    Your task is to clearly and concisely justify the model's classification decision.
    Instructions:
    1. you will be given a list of key words that were most influential for the classification decision.
    2. your answer should be an explanation for the text record, indicating why the model made a particular decision based on these words.
    3 Begin by stating what the classification was.
    4. explain how the keywords contributed to this decision, linking them to the topic of the message.
    5. the text should be 1-2 sentences long.

    Input format:
    - Classification: {class_name}
    - Text: {text}
    - Key words: {top_words}.
    """
    
    prompt = prompt_base.format(
        class_name=record['predicted_label'],
        text=record['text'],
        top_words=', '.join(top_k_words)
    )
    print(prompt)
    
    client = OpenAI(
    api_key=api_key,
    base_url=base_url
    )
    response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "system", "content": "You are an expert in Explainable Artificial Intelligence (XAI)."},
                    {"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
    
    response_text = response.choices[0].message.content.strip()
    
    return response_text

prompt_etp = """
    ### Instructions:
    - Carefully examine the input and assign a numeric **importance score (from 0 to 1)** to each word in the text, reflecting how much it contributes to your classification decision.
    - Based on your analysis, predict a single category. The category must be one of the numeric values.
    - Provide your **confidence** in the prediction as a float from 0 to 1.
    - **Return your results strictly as a JSON object and include nothing else in your response.**

    Return your results strictly as a JSON object with the following structure:
    {{
        "word_importance":
            ["word1", 0.4],
            ["word2", 0.8],
            ["word3", 0.0],
            ...
        ,
        "category": "CATEGORY_NAME",
        "confidence": 0.87
    }}
    """
    
    
def calculate_explanation(text, top_logrprobs, prompt, all_labels):
    record = {
        'text': text,
        'predicted_label': all_labels[top_logrprobs[0]['token']],  
        'top_logrprobs': top_logrprobs,
        'probabilities': get_logprobs(top_logrprobs, all_labels)
    }
    
    logit_bias = {}
    for i in range(0, len(all_labels.keys())):
        logit_bias[str(48 + i)] = 100

    SHAP_explanation = get_SHAP_values(text, prompt, logit_bias, all_labels, API_KEY, BASE_URL)
    
    occlusion_explanation = get_occlusion_values(text, prompt, logit_bias, all_labels, API_KEY, BASE_URL, record['probabilities'], original_class=record['predicted_label'])
    
    self_explanation = get_self_explanations(text, prompt + prompt_etp, all_labels, API_KEY, BASE_URL)
    
    top_k_words = get_top_k_words_combined(occlusion_explanation, 
                                            SHAP_explanation,
                                            self_explanation['words_probability'],
                                            k_ratio=0.2,
                                            shap_weight=2)

    explanation = get_final_explanation(top_k_words, record, API_KEY, BASE_URL)
    
    return explanation

