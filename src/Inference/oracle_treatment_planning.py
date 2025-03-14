import re
import json
import time
import random
import requests
import numpy as np
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

# ======================
# Configuration Constants
# ======================

# API Keys and URLs
QWQ_URL = 'https://api.siliconflow.cn/v1/'
QWQ_API_KEY = 'YOUR_API_KEY'

GEMINI_URL = 'https://aigptapi.com/v1/'
GEMINI_API_KEY = 'YOUR_API_KEY'

O1_API_KEY_LIST = [
    "sk-oJTcF42OtAkjkA2MCFVXjVLGJLghrCPJ8a9XIJ1JE0NoYVmb",
    "YOUR_API_KEY",
]

DEEPSEEK_R1_URL = "http://10.17.3.65:1025/v1/chat/completions"

# Path constants
DATA_PATH = '../../data/MedRBench/treatment_496_cases_with_rare_disease_165.json'
TREATMENT_PROMPT_PATH = './treatment_plan_prompt.txt'
OUTPUT_DIR = "oracle_treatment_plan"

# Default settings
DEFAULT_SYSTEM_PROMPT = "You are a professional doctor"

# ======================
# Utility Functions
# ======================

def load_instruction(txt_path):
    """Load prompt template from file"""
    try:
        with open(txt_path, encoding='utf-8') as fp:
            return fp.read()
    except Exception as e:
        print(f"Error loading prompt template from {txt_path}: {e}")
        return None

def ensure_output_dir(directory):
    """Ensure output directory exists"""
    import os
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created output directory: {directory}")

def save_results(data, filename):
    """Save results to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

def load_data():
    """Load data"""
    # Load full dataset
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        json_datas = json.load(f)
    return json_datas
    

# ======================
# Model API Interfaces
# ======================

def query_qwq_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the QwQ model and handle potential rate limit errors"""
    client = OpenAI(
        base_url=QWQ_URL,
        api_key=QWQ_API_KEY
    )
    
    while True:
        try:
            response = client.chat.completions.create(
                model="Qwen/QwQ-32B-Preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                stream=False, 
                max_tokens=4096
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            error_message = str(e)
            if '429' in error_message:
                print(f"Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
                print('Retrying...')
            else:
                print(f"Error querying QwQ model: {e}")
                return None

def query_o1_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the O1/O3-mini model with round-robin API key selection"""
    max_try_times = 5
    curr_try = 0
    
    while curr_try < max_try_times:
        try:
            client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key=random.choice(O1_API_KEY_LIST)
            )
            completion = client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            curr_try += 1
            if curr_try >= max_try_times:
                print(f"Error querying O1 model: {e}")
                return "Error."
            time.sleep(5)

def query_gemini_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query the Gemini model and handle rate limits"""
    for retry in range(3):
        try:
            client = OpenAI(
                base_url=GEMINI_URL,
                api_key=GEMINI_API_KEY
            )
            response = client.chat.completions.create(
                model="gemini-2.0-flash-thinking-exp-01-21",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                stream=False, 
                max_tokens=4096
            )
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(f"Error on attempt {retry+1}/3: {e}")
            time.sleep(5)
    
    print("Failed to query Gemini model after 3 attempts")
    return None

def query_deepseek_r1_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query DeepSeek-R1 model via direct HTTP request"""
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "DeepSeek-R1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ],
        "temperature": 0.6,
        "stream": False,
        "max_tokens": 10000,
    }
    
    max_retry = 3
    curr_retry = 0
    
    while curr_retry < max_retry:
        try:
            response = requests.post(DEEPSEEK_R1_URL, json=data, headers=headers)
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # DeepSeek-R1 provides reasoning in <think> tags
            reasoning = content.split('</think>')[0].replace('<think>', '').strip()
            answer = content.split('</think>')[1].strip()
            
            return reasoning, answer
        except Exception as e:
            curr_retry += 1
            print(f"Error: {e}, retrying... {curr_retry}/{max_retry}")
            time.sleep(2)
    
    return None, None

def query_baichuan_model(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT, model=None, tokenizer=None):
    """Query the Baichuan model using transformers library"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate text
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated text
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    except Exception as e:
        print(f"Error querying Baichuan model: {e}")
        return None

# ======================
# Data Processing Functions
# ======================

def process_qwq_data(data_id, data, prompt_template):
    """Process a single data item with QwQ model"""    
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        
        response = query_qwq_model(prompt)
        result['content'] = response
        
        return data_id, result
    except Exception as e:
        print(f"Error processing QwQ data {data_id}: {e}")
        return data_id, None

def process_o1_data(data_id, data, prompt_template):
    """Process a single data item with O1/O3-mini model"""
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        
        response = query_o1_model(prompt)
        result['content'] = response
        
        return data_id, result
    except Exception as e:
        print(f"Error processing O1 data {data_id}: {e}")
        return data_id, None

def process_gemini_data(data_id, data, prompt_template):
    """Process a single data item with Gemini model"""
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        
        response = query_gemini_model(prompt)
        result['content'] = response
        
        return data_id, result
    except Exception as e:
        print(f"Error processing Gemini data {data_id}: {e}")
        return data_id, None

def process_deepseek_r1_data(data_id, data, prompt_template):
    """Process a single data item with DeepSeek-R1 model"""
    try:
        result = {}
        patient_case = data['generate_case']['case_summary']
        prompt = prompt_template.format(case=patient_case)
        
        reasoning, answer = query_deepseek_r1_model(prompt)
        result['reasoning'] = reasoning
        result['content'] = answer
        
        return data_id, result
    except Exception as e:
        print(f"Error processing DeepSeek-R1 data {data_id}: {e}")
        return data_id, None

# ======================
# Main Inference Functions
# ======================

def run_inference_with_model(model_name, process_func, max_workers=8):
    """Generic function to run inference with any model"""
    print(f"Running treatment inference with {model_name}")
    
    # Load prompt template
    prompt_template = load_instruction(TREATMENT_PROMPT_PATH)
    if not prompt_template:
        print(f"Error: Failed to load prompt template")
        return
    
    # Load data
    datas = load_data()
    if not datas:
        print(f"Error: No data to process")
        return
    
    # Ensure output directory exists
    ensure_output_dir(OUTPUT_DIR)
    
    # Process data with the specified model using concurrent processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for data_id, data in datas.items():
            futures.append(executor.submit(process_func, data_id, data, prompt_template))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing with {model_name}"):
            data_id, result = future.result()
            if result is not None:
                datas[data_id][model_name.lower()] = result

    # Save results
    filename = f"{OUTPUT_DIR}/{model_name.lower()}_all_output.json"
    save_results(datas, filename)

def inference_qwq():
    """Run treatment inference with QwQ model"""
    run_inference_with_model("qwq", process_qwq_data, max_workers=8)

def inference_o1():
    """Run treatment inference with O1/O3-mini model"""
    run_inference_with_model("o3-mini", process_o1_data, max_workers=8)

def inference_gemini():
    """Run treatment inference with Gemini model"""
    run_inference_with_model("gemini", process_gemini_data, max_workers=8)

def inference_deepseek_r1():
    """Run treatment inference with DeepSeek-R1 model"""
    run_inference_with_model("r1", process_deepseek_r1_data, max_workers=8)

def inference_baichuan():
    """Run treatment inference with Baichuan model directly (not using concurrency)"""
    print("Running treatment inference with Baichuan model")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Load model
        model_name = "Baichuan-M1-14B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='cuda:0',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Load prompt template
        prompt_template = load_instruction(TREATMENT_PROMPT_PATH)
        if not prompt_template:
            print(f"Error: Failed to load prompt template")
            return
        
        # Load data
        datas = load_data()
        if not datas:
            print(f"Error: No data to process")
            return
        
        # Ensure output directory exists
        ensure_output_dir(OUTPUT_DIR)
        
        # Process data
        for key in tqdm(datas.keys(), desc="Processing with Baichuan"):
              
            patient_case = datas[key]['generate_case']['case_summary']
            prompt = prompt_template.format(case=patient_case)
            
            response = query_baichuan_model(prompt, model=model, tokenizer=tokenizer)
            if response:
                if 'baichuan' not in datas[key]:
                    datas[key]['baichuan'] = {}
                datas[key]['baichuan']['content'] = response
        
        # Save results
        filename = f"{OUTPUT_DIR}/baichuan_all_output.json"
        save_results(datas, filename)
        
    except Exception as e:
        print(f"Error in Baichuan inference: {e}")

if __name__ == "__main__":
    # Run inference with all models
    inference_deepseek_r1()
    inference_qwq()
    inference_o1()
    inference_gemini()
    # Only run this if you have the Baichuan model and GPU support
    inference_baichuan()