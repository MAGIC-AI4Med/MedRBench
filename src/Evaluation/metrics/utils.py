from openai import OpenAI
import random
import re
import json
from typing import List, Optional

GPT_KEY_FILE = 'gpt_key.txt'

def workflow(model_name, instruction, input_text):
    """Execute a single API call to evaluate content"""
    with open(GPT_KEY_FILE, 'r') as f:
        api_keys = f.readlines()
    selected_key = random.choice(api_keys).strip()
    
    client = OpenAI(
        base_url="https://api.gpts.vin/v1",
        api_key=selected_key
    )

    completion = client.chat.completions.create(
        model = model_name,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]
    )
    return completion.choices[0].message.content

def workflow_multi_turn(model_name, input_text, history_messages):
    with open(GPT_KEY_FILE, 'r') as f:
        api_keys = f.readlines()
    selected_key = random.choice(api_keys).strip()
    
    client = OpenAI(
        base_url="https://api.gpts.vin/v1",
        api_key=selected_key
    )
    history_messages.append({"role": "user", "content": input_text})
    completion = client.chat.completions.create(
        model=model_name,
        messages=history_messages
    )
    return completion.choices[0].message.content


def load_instruction(file_path):
    """Load instruction text from file.
    
    Args:
        file_path: Path to the instruction template file
        
    Returns:
        String containing the instruction template
    """
    with open(file_path, 'r', encoding='utf-8') as fp:
        return fp.read()
    
    
def safe_json_parse(model_output, retry_count=0):
    """Safely parse JSON and handle formatting errors.
    
    Args:
        model_output: JSON string to parse
        retry_count: Current retry attempt number
        
    Returns:
        Parsed JSON object or None if parsing fails after max retries
    """
    max_retries = 3
    if retry_count >= max_retries:
        print("JSON parse error after maximum retries")
        return None
    try:
        parsed_output = json.loads(model_output)
        return parsed_output
    except json.JSONDecodeError as e:
        corrected_output = request_correction_from_model(model_output, str(e), retry_count)
        return safe_json_parse(corrected_output, retry_count+1)


def request_correction_from_model(incorrect_output, error_message, retry_count, model_name = 'gpt-4o-2024-11-20'):
    """Request model to fix JSON formatting errors.
    
    Args:
        incorrect_output: Malformed JSON string
        error_message: Error message from JSON decoder
        retry_count: Current retry attempt number
        
    Returns:
        Corrected JSON string
    """
    max_retries = 3
    if retry_count >= max_retries:
        return incorrect_output
    
    system_prompt = 'You are a JSON format modifier.'
    input_text = f"Fixed the following output JSON format error, ensure that it is a valid JSON string, and the current error message is{error_message}\
          only output the correct JSON string that can be parsed, do not output other content:\n{incorrect_output}"
    
    corrected_completion = workflow(
        model_name=model_name, 
        instruction=system_prompt, 
        input_text=input_text
    ).replace('```json', '').replace('```', '').strip()
    
    print(f'Try correct {retry_count}\n before:\n{incorrect_output}\nafter:\n{corrected_completion}')
    
    try:
        output = json.loads(corrected_completion)
        return json.dumps(output)
    except json.JSONDecodeError as e:
        return request_correction_from_model(corrected_completion, str(e), retry_count + 1)

