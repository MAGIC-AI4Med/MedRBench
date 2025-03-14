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

ASK_TEMPLATE_PATH = f'instructions/1turn_prompt_examination_recommend.txt'
FINAL_TEMPLATE_PATH = f'instructions/1turn_prompt_make_diagnosis.txt'
GPT_PROMPT_PATH = f'instructions/patient_agent_prompt.txt'

# Default settings
DEFAULT_SYSTEM_PROMPT = "You are a professional doctor"
VERBOSE = False

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

def parse_assessment_output(answer_text):
    """Extract conclusion and additional info request from model response"""
    pattern = r'### Conclusion:\s*(.*?)\s*### Additional Information Required:\s*(.*)'
    matches = re.search(pattern, answer_text, re.DOTALL)
    if matches:
        preliminary_conclusion = matches.group(1).strip()
        additional_info_required = matches.group(2).strip()
        return preliminary_conclusion, additional_info_required
    else:
        raise ValueError("Could not parse answer format - missing expected sections")

def ensure_output_dir(directory):
    """Ensure output directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created output directory: {directory}")

# ======================
# Model API Interfaces
# ======================

def gpt4o_workflow(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query GPT-4o model for additional information retrieval"""
    max_retry = 3
    curr_retry = 0
    
    while curr_retry < max_retry:
        try:
            # You should provide your own API keys in a production environment
            # This is just a placeholder
            client = OpenAI(
                base_url="https://api.gpts.vin/v1",
                api_key=random.choice(O1_API_KEY_LIST) if O1_API_KEY_LIST else "your-api-key"
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
    """Query QwQ model with chain-of-thought support"""
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
            content = response.choices[0].message.content
            
            # Parse reasoning and answer from content
            if "### Conclusion" in content:
                parts = content.split('### Conclusion')
                reasoning = parts[0].strip()
                answer = "### Conclusion" + parts[1].strip()
                return answer, reasoning
            else:
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
                return content, ""
            
        except Exception as e:
            curr_try += 1
            if curr_try >= max_try_times:
                print(f"Error querying O1 model: {e}")
                return None, None
            time.sleep(5)

def gemini_workflow(messages):
    """Query Gemini model with chain-of-thought support"""
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
            content = response.choices[0].message.content
            content = content.replace('```', '')
            
            # Parse reasoning and answer
            if "### Conclusion" in content:
                parts = content.split('### Conclusion')
                reasoning = parts[0].strip()
                answer = "### Conclusion" + parts[1].strip()
                return answer, reasoning
            else:
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

def deepseek_r1_workflow(input_text, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Query DeepSeek-R1 model via direct HTTP request"""
    if isinstance(input_text, list):  # Handle message list format
        messages = input_text
    else:  # Handle text format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "DeepSeek-R1",
        "messages": messages,
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
        
        # Parse reasoning and answer
        if "### Conclusion:" in response:
            reasoning = response.split('### Conclusion:')[0].strip()
            answer = '### Conclusion:' + response.split('### Conclusion:')[1].strip()
            return answer, reasoning
        else:
            return response, ""
    except Exception as e:
        print(f"Error in Baichuan inference: {e}")
        return None, None

# ======================
# Model Processing Functions
# ======================

def process_instance(key, json_data, gpt_prompt, ask_template, final_template, model_name, **kwargs):
    """
    Generic function to process a single case with any model
    
    Parameters:
    -----------
    key : str
        Case identifier
    json_data : dict
        Dictionary containing all case data
    gpt_prompt : str
        Template for GPT-4o prompt
    ask_template : str
        Template for initial query to primary model
    final_template : str
        Template for final query to primary model
    model_name : str
        Name of the primary model to use
    **kwargs : dict
        Additional model-specific parameters
    """
    # Define output path based on model name
    output_dir = f'1_turn_{model_name.lower()}'
    output_file = f'{output_dir}/log_{key}.json'
    
    # Skip if already processed
    if os.path.exists(output_file):
        return
    
    # Configure model-specific functions
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
    
    # Retry loop for robustness
    for try_idx in range(3):
        try:    
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
            gpt_instruction = gpt_prompt.format(case=case_summary_without_ancillary_test, ancillary_test_results=ancillary_test)
            
            # Initial messages
            primary_messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": ask_template.format(case=case_summary_without_ancillary_test)}
            ]
            
            # Log messages with reasoning separately
            messages_log = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": ask_template.format(case=case_summary_without_ancillary_test)}
            ]
            
            # Step 1: Get preliminary diagnosis and questions from primary model
            primary_answer, primary_reasoning = model_workflow(primary_messages)
                
            if VERBOSE:
                print(f"Primary model reasoning:\n{primary_reasoning}")
                print(f"Primary model answer:\n{primary_answer}")
                
            if not primary_answer:
                print(f"Error: No response from primary model")
                continue
                
            # Clean up response and extract information requests
            primary_answer = primary_answer.replace('```', '').strip()
            preliminary_conclusion, additional_info_required = parse_assessment_output(primary_answer)
            
            # Update message history
            primary_messages.append({"role": "assistant", "content": primary_answer})
            messages_log.append({"role": "assistant", "content": {
                'reasoning': primary_reasoning, 
                'answer': primary_answer
            }})
            
            # Step 2: Use GPT-4o to answer requested additional information
            gpt_input = f"The junior physician wants the following information:\n{additional_info_required}"
            gpt_response = gpt4o_workflow(gpt_input, gpt_instruction)
            
            if VERBOSE:
                print(f"GPT-4o response:\n{gpt_response}")
                
            if not gpt_response:
                print(f"Error: No response from GPT-4o")
                continue
                
            # Format response for primary model
            formatted_response = final_template.format(additional_information=gpt_response)
            
            # Update message history
            primary_messages.append({"role": "user", "content": formatted_response})
            messages_log.append({"role": "user", "content": formatted_response})
            
            # Step 3: Get final diagnosis from primary model with additional information
            if model_name == "deepseekr1":
                final_answer, final_reasoning = model_workflow(primary_messages)
            else:
                final_answer, final_reasoning = model_workflow(primary_messages)
                
            if VERBOSE:
                print(f"Final answer:\n{final_answer}")
                
            if not final_answer:
                print(f"Error: No final response from primary model")
                continue
                
            # Clean up response
            final_answer = final_answer.replace('```', '').strip()
            
            # Update message history
            primary_messages.append({"role": "assistant", "content": final_answer})
            messages_log.append({"role": "assistant", "content": {
                'reasoning': final_reasoning, 
                'answer': final_answer
            }})
            
            # Prepare output data
            output_messages = []
            for msg in messages_log:
                output_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            
            log_data = {
                'output_messages': output_messages,
            }
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as fp:
                json.dump(log_data, fp, ensure_ascii=False, indent=4)
                
            print(f"Successfully processed {key} with {model_name}")
            return
            
        except Exception as e:
            if try_idx == 2:  # If final retry
                print(f"Error processing {key} with {model_name}: {e}")
                error_log = f'level2_{model_name.lower()}_error.log'
                with open(error_log, 'a') as fp:
                    fp.write(f"Error: {e}, {key}\n")

# ======================
# Main Inference Functions
# ======================

def run_inference(model_name, max_workers=8, **kwargs):
    """
    Run inference for a specific model
    
    Parameters:
    -----------
    model_name : str
        Name of the model to use
    max_workers : int
        Number of parallel workers
    **kwargs : dict
        Additional model-specific parameters
    """
    print(f"Running 1-turn inference with {model_name}")
    
    # Load templates
    ask_template = load_instruction(ASK_TEMPLATE_PATH)
    final_template = load_instruction(FINAL_TEMPLATE_PATH)
    gpt_prompt = load_instruction(GPT_PROMPT_PATH)
    
    if not all([ask_template, final_template, gpt_prompt]):
        print("Error: Failed to load required templates")
        return
    
    # Load case data
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as fp:
            json_data = json.load(fp)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
            
    # Create output directory
    output_dir = f'1_turn_{model_name.lower()}'
    ensure_output_dir(output_dir)
    
    # Special handling for Baichuan which doesn't use multiprocessing
    keys = list(json_data.keys())
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
                process_instance(key, json_data, gpt_prompt, ask_template, final_template, 
                                model_name, model=model, tokenizer=tokenizer)
        except Exception as e:
            print(f"Error initializing Baichuan model: {e}")
        return
    else:
        # Process cases with multiprocessing for other models
        with Pool(processes=max_workers) as pool:
            results = pool.starmap(
                process_instance, 
                [(key, json_data, gpt_prompt, ask_template, final_template, model_name) for key in keys]
            )
            
            # Show progress with tqdm
            list(tqdm.tqdm(results, total=len(keys), desc=f"Processing with {model_name}"))

def inference_qwq():
    """Run inference with QwQ model"""
    run_inference("qwq", max_workers=8)

def inference_o1():
    """Run inference with O1 (O3-mini) model"""
    run_inference("o1", max_workers=8)

def inference_gemini():
    """Run inference with Gemini model"""
    run_inference("gemini", max_workers=8)

def inference_deepseekr1():
    """Run inference with DeepSeek-R1 model"""
    run_inference("deepseekr1", max_workers=8)

def inference_baichuan():
    """Run inference with Baichuan model"""
    run_inference(
        "baichuan", 
        max_workers=0,  # Baichuan uses sequential processing
        model_path="Baichuan-M1-14B-Instruct"
    )

if __name__ == '__main__':
    # Run inference with all models
    inference_qwq()
    inference_o1()
    inference_gemini()
    inference_deepseekr1()
    # Only run this if you have the Baichuan model and GPU support
    inference_baichuan()