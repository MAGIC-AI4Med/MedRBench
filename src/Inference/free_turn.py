import os
import json
import re
import time
import random
import requests
import numpy as np
import tqdm
from multiprocessing import Pool
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
DATA_PATH = '../../data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json'
INSTRUCTION_DIR = "instructions"
INITIAL_TEMPLATE_PATH = f'{INSTRUCTION_DIR}/free_turn_first_turn_prompt.txt'
PROCESS_TEMPLATE_PATH = f'{INSTRUCTION_DIR}/free_turn_following_turn_prompt.txt'
GPT_PROMPT_PATH = f'{INSTRUCTION_DIR}/patient_agent_prompt.txt'

# Default settings
DEFAULT_SYSTEM_PROMPT = "You are a professional doctor"
VERBOSE = False
MAX_TURNS = 5  # Maximum number of diagnostic turns

# ======================
# Utility Functions
# ======================

def load_instruction(txt_path):
    """Load prompt template from file"""
    try:
        with open(txt_path) as fp:
            return fp.read()
    except Exception as e:
        print(f"Error loading instruction from {txt_path}: {e}")
        return None

def parse_deepseekr1_answer(answer_text):
    """Extract additional info request and conclusion from model response"""
    print(answer_text)  # Debug output
    pattern = r'### Additional Information Required:\s*(.*?)\s*### Conclusion:\s*(.*)'
    matches = re.search(pattern, answer_text, re.DOTALL)
    if matches:
        additional_info_required = matches.group(1).strip()
        preliminary_conclusion = matches.group(2).strip()
        return preliminary_conclusion, additional_info_required
    else:
        raise ValueError("Could not parse answer format - missing expected sections")

def ensure_output_dir(directory):
    """Ensure output directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created output directory: {directory}")

# ======================
# Model API Interfaces
# ======================

def gpt4o_workflow(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query GPT-4o model for additional information retrieval"""
    max_retry = 5
    curr_retry = 0
    
    while curr_retry < max_retry:
        try:
            client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key=random.choice(O1_API_KEY_LIST)
            )
            completion = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            curr_retry += 1
            print(f"Error ({curr_retry}/{max_retry}): {e}")
            time.sleep(5)
    
    return None

def qwq_workflow(messages):
    """Query QwQ model"""
    client = OpenAI(
        base_url=QWQ_URL,
        api_key=QWQ_API_KEY
    )
    
    while True:
        try:
            response = client.chat.completions.create(
                model="Qwen/QwQ-32B-Preview",
                messages=messages, 
                stream=False, 
                max_tokens=8192
            )
            content = response.choices[0].message.content.replace('```', '').strip()
            # In level3, QwQ doesn't provide separate reasoning
            return content, ""
            
        except Exception as e:
            error_message = str(e)
            if '429' in error_message:
                print(f"Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
                print('Retrying...')
            else:
                print(f"Error querying QwQ: {e}")
                return None, None

def o1_workflow(messages):
    """Query O1/O3-mini model with chain-of-thought support"""
    max_try_times = 3
    curr_try = 0
    
    while curr_try < max_try_times:
        try:
            client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key=random.choice(O1_API_KEY_LIST)
            )
            completion = client.chat.completions.create(
                model="o3-mini",
                messages=messages,
            )
            content = completion.choices[0].message.content
            
            # Parse reasoning and answer
            if "### Chain of Thought" in content:
                reasoning = content.split('### Chain of Thought')[0].strip()
                answer = "### Chain of Thought" + content.split('### Chain of Thought')[1].strip()
                return answer.replace('```', '').strip(), reasoning.replace('```', '').strip()
            else:
                return content.replace('```', '').strip(), ""
            
        except Exception as e:
            curr_try += 1
            if curr_try >= max_try_times:
                print(f"Error querying O1 model: {e}")
                return None, None
            time.sleep(5)

def gemini_workflow(messages):
    """Query Gemini model"""
    client = OpenAI(
        base_url=GEMINI_URL,
        api_key=GEMINI_API_KEY
    )
    
    while True:
        try:
            response = client.chat.completions.create(
                model="gemini-2.0-flash-thinking-exp-01-21",
                messages=messages, 
                stream=False, 
                max_tokens=8192
            )
            content = response.choices[0].message.content.replace('```', '').strip()
            # In level3, Gemini doesn't provide separate reasoning
            return content, ""
            
        except Exception as e:
            error_message = str(e)
            print(f"Error: {e}")
            if '429' in error_message:
                print(f"Rate limit exceeded. Waiting for 30 seconds...")
                time.sleep(30)
                print('Retrying...')
            else:
                return None, None

def deepseek_r1_workflow(messages):
    """Query DeepSeek-R1 model via direct HTTP request"""
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "DeepSeek-R1",
        "messages": messages,
        "temperature": 0.6,
        "stream": False,
        "max_tokens": 12800,
    }
    
    max_retry = 3
    curr_retry = 0
    
    while curr_retry < max_retry:
        try:
            response = requests.post(DEEPSEEK_R1_URL, json=data, headers=headers)
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse reasoning (in <think> tags) and answer
            if "</think>" in content:
                reasoning = content.split('</think>')[0].replace('<think>', '').strip()
                answer = content.split('</think>')[1].strip()
                return answer, reasoning
            else:
                return content, ""
            
        except Exception as e:
            curr_retry += 1
            print(f"Error: {e}, retrying... {curr_retry}/{max_retry}")
            time.sleep(2)
    
    return None, None

def baichuan_workflow(messages, model, tokenizer):
    """Query Baichuan model using HuggingFace transformers"""
    try:
        import torch
        
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
        # In level3, Baichuan doesn't provide separate reasoning
        return response.replace('```', '').strip(), ""
    except Exception as e:
        print(f"Error in Baichuan inference: {e}")
        return None, None

# ======================
# Core Multi-Turn Inference Process
# ======================

def process_instance(key, json_data, gpt_prompt, initial_template, process_template, model_name, **kwargs):
    """
    Process a single case with multi-turn interaction using the specified model
    
    Parameters:
    -----------
    key : str
        Case identifier
    json_data : dict
        Dictionary containing all case data
    gpt_prompt : str
        Template for GPT-4o prompt
    initial_template : str
        Template for initial query to primary model
    process_template : str
        Template for subsequent queries to primary model
    model_name : str
        Name of the primary model to use
    **kwargs : dict
        Additional model-specific parameters
    """
    # Define output path based on model name
    output_dir = f'level3/{model_name.lower()}'
    output_file = f'{output_dir}/log_{key}.json'
    
    # Skip if already processed
    if os.path.exists(output_file):
        return
    
    # Configure model-specific function
    if model_name == "qwq":
        model_workflow = qwq_workflow
    elif model_name == "o1":
        model_workflow = o1_workflow 
    elif model_name == "gemini":
        model_workflow = gemini_workflow
    elif model_name == "deepseekr1":
        model_workflow = deepseek_r1_workflow
    elif model_name == "baichuan":
        if 'model' not in kwargs or 'tokenizer' not in kwargs:
            print(f"Error: Baichuan requires model and tokenizer objects")
            return
        model_workflow = lambda msgs: baichuan_workflow(msgs, kwargs['model'], kwargs['tokenizer'])
    else:
        print(f"Error: Unknown model type '{model_name}'")
        return
    
    try:
        # Get case data
        one_instance = json_data[key]
        case_summary = one_instance['generate_case']['case_summary']
        
        if "Ancillary Tests" in case_summary:
            case_summary_paragrapgh = case_summary.strip().split('\n')
            for idx in range(len(case_summary_paragrapgh)):
                if "Ancillary Tests" in case_summary_paragrapgh[idx]:
                    case_summary_without_ancillary_test = "\n".join(case_summary_paragrapgh[:idx])
                    ancillary_test = "\n".join(case_summary_paragrapgh[idx:])
                    break
    
        # Prepare prompts
        gpt_instruction = gpt_prompt.format(
            case=case_summary_without_ancillary_test, 
            ancillary_test_results=ancillary_test
        )
        
        # Initial messages
        primary_messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": initial_template.format(case=case_summary_without_ancillary_test)}
        ]
        
        # Log messages with reasoning separately
        messages_log = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": initial_template.format(case=case_summary_without_ancillary_test)}
        ]
        
        # Multi-turn interaction loop
        turn_idx = 1
        while turn_idx <= MAX_TURNS:
            # Get model response for current state
            primary_answer, primary_reasoning = model_workflow(primary_messages)
            
            if VERBOSE:
                print(f"Turn {turn_idx} - Primary model reasoning:\n{primary_reasoning}")
                print(f"Turn {turn_idx} - Primary model answer:\n{primary_answer}")
                
            if not primary_answer:
                print(f"Error: No response from primary model on turn {turn_idx}")
                break
                
            # Clean up response and extract additional information request
            primary_answer = primary_answer.replace('```', '').strip()
            
            # Update message history
            primary_messages.append({"role": "assistant", "content": primary_answer})
            messages_log.append({"role": "assistant", "content": {
                'reasoning': primary_reasoning, 
                'answer': primary_answer
            }})
            
            # Parse response to get conclusion and additional info request
            try:
                preliminary_conclusion, additional_info_required = parse_deepseekr1_answer(primary_answer)
            except ValueError as e:
                print(f"Error parsing model response: {e}")
                break
                
            # Check if we're done with information gathering
            if "Not required" in additional_info_required or turn_idx == MAX_TURNS:
                break
                
            # Request additional information from GPT-4o
            gpt_input = f"The junior physician wants the following information:\n{additional_info_required}"
            gpt_response = gpt4o_workflow(gpt_input, gpt_instruction)
            
            if VERBOSE:
                print(f"Turn {turn_idx} - GPT-4o response:\n{gpt_response}")
                
            if not gpt_response:
                print(f"Error: No response from GPT-4o on turn {turn_idx}")
                break
                
            # Format response for next turn
            formatted_response = process_template.format(additional_information=gpt_response)
            
            # Add final turn warning if needed
            if turn_idx == MAX_TURNS - 1:
                formatted_response = "In the next turn, you cannot ask any additional infomation and must make a final diagnoisis.\n" + formatted_response
            
            # Update message history
            primary_messages.append({"role": "user", "content": formatted_response})
            messages_log.append({"role": "user", "content": formatted_response})
            
            # Increment turn counter
            turn_idx += 1
            
        # Prepare output messages
        output_messages = []
        for msg in messages_log:
            output_messages.append({
                'role': msg['role'],
                'content': msg['content'],
            })
        
        # Prepare output data
        log_data = {
            'deepseek_messages': output_messages,  # Kept name for compatibility
            'ground_truth': one_instance['final_diagnosis'],
            'ancillary_test_results': one_instance['level2']['ancillary_test'],
            'turns': turn_idx,
        }
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as fp:
            json.dump(log_data, fp, ensure_ascii=False, indent=4)
            
        print(f"Successfully processed {key} with {model_name} in {turn_idx} turns")
        
    except Exception as e:
        print(f"Error processing {key} with {model_name}: {e}")
        error_log = f'level3/{model_name.lower()}.log'
        with open(error_log, 'a') as fp:
            fp.write(f"Error: {e}, {key}\n")

def safe_process_instance(key, json_data, gpt_prompt, initial_template, process_template, model_name, **kwargs):
    """Wrapper function with retry logic for robustness"""
    output_dir = f'level3/{model_name.lower()}'
    output_file = f'{output_dir}/log_{key}.json'
    
    # Skip if already processed
    if os.path.exists(output_file):
        return
    
    # Try up to 3 times to process the instance
    for try_idx in range(3):
        try:
            process_instance(key, json_data, gpt_prompt, initial_template, process_template, model_name, **kwargs)
            return  # Success, exit the retry loop
        except Exception as e:
            if try_idx == 2:  # If final retry
                print(f"Error processing {key} with {model_name} after 3 attempts: {e}")
                error_log = f'level3/{model_name.lower()}.log'
                with open(error_log, 'a') as fp:
                    fp.write(f"Error: {e}, {key}\n")

# ======================
# Main Inference Functions
# ======================

def run_inference(model_name, max_workers=8, **kwargs):
    """
    Run multi-turn inference for a specific model
    
    Parameters:
    -----------
    model_name : str
        Name of the model to use
    max_workers : int
        Number of parallel workers
    **kwargs : dict
        Additional model-specific parameters
    """
    print(f"Running Level 3 (multi-turn) inference with {model_name}")
    
    # Load templates
    initial_template = load_instruction(INITIAL_TEMPLATE_PATH)
    process_template = load_instruction(PROCESS_TEMPLATE_PATH)
    gpt_prompt = load_instruction(GPT_PROMPT_PATH)
    
    if not all([initial_template, process_template, gpt_prompt]):
        print("Error: Failed to load required templates")
        return
    
    # Load case data
    with open(DATA_PATH, 'r', encoding='utf-8') as fp:
        json_data = json.load(fp)
    keys = list(json_data.keys())
    print(f"Processing {len(keys)} cases with {model_name}")
    
    # Create output directory
    output_dir = f'level3/{model_name.lower()}'
    ensure_output_dir(output_dir)
    
    # Special handling for Baichuan which doesn't use multiprocessing
    if model_name.lower() == "baichuan":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Load model
            model_path = kwargs.get('model_path', "Baichuan-M1-14B-Instruct")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map='cuda:0',
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # Process cases sequentially
            for key in tqdm.tqdm(keys, desc=f"Processing with {model_name}"):
                safe_process_instance(
                    key, 
                    json_data, 
                    gpt_prompt, 
                    initial_template, 
                    process_template, 
                    model_name, 
                    model=model, 
                    tokenizer=tokenizer
                )
        except Exception as e:
            print(f"Error initializing Baichuan model: {e}")
        return
    
    # Process cases with multiprocessing for other models
    with Pool(processes=max_workers) as pool:
        results = pool.starmap(
            safe_process_instance, 
            [(key, json_data, gpt_prompt, initial_template, process_template, model_name) for key in keys]
        )
        
        # Show progress with tqdm
        list(tqdm.tqdm(results, total=len(keys), desc=f"Processing with {model_name}"))

def inference_deepseek_r1():
    """Run multi-turn inference with DeepSeek-R1 model"""
    run_inference("r1", max_workers=8)

def inference_qwq():
    """Run multi-turn inference with QwQ model"""
    run_inference("qwq", max_workers=8)

def inference_o1():
    """Run multi-turn inference with o3-mini model"""
    run_inference("o1", max_workers=8)

def inference_gemini():
    """Run multi-turn inference with Gemini model"""
    run_inference("gemini", max_workers=8)

def inference_baichuan():
    """Run multi-turn inference with Baichuan model"""
    run_inference(
        "baichuan", 
        max_workers=0,  # Baichuan uses sequential processing
        model_path="Baichuan-M1-14B-Instruct"
    )

if __name__ == '__main__':
    # Run inference with desired models
    inference_deepseek_r1()
    inference_qwq()
    inference_o1()
    inference_gemini()
    # Only run this if you have the Baichuan model and GPU support
    inference_baichuan()