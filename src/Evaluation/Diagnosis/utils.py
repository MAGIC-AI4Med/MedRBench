from openai import OpenAI
import random
import re
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



def split_reasoning_gemini(content: str) -> List[str]:
    """Split Gemini model's reasoning into steps.
    
    Args:
        content: Raw model output text
        
    Returns:
        List of reasoning steps
    """
    try:
        # Extract reasoning part
        if '### Answer:' in content:
            reasoning = content.split('### Answer:')[0]
        else:
            reasoning = content
            
        # Extract steps using regex pattern
        pattern = r"<step\s+(\d+)>\s*(.*?)(?=\n<step\s+\d+>|$)"
        matches = re.findall(pattern, reasoning, re.DOTALL)
        
        # Extract just the step content
        reasoning_steps = [step_content.strip() for _, step_content in matches]
        
        return reasoning_steps
    except Exception as e:
        print(f"Error splitting Gemini reasoning: {e}")
        return []


def split_reasoning_r1(content: str) -> List[str]:
    """Split DeepSeek-R1 model's reasoning into steps.
    
    Args:
        content: Raw model output text
        
    Returns:
        List of reasoning steps
    """
    try:
        # Extract reasoning part
        if '### Answer:' in content:
            reasoning = content.split('### Answer:')[0]
        else:
            reasoning = content
            
        # Extract steps using regex pattern
        pattern = r"<step\s+(\d+)>\s*(.*?)(?=\n<step\s+\d+>|$)"
        matches = re.findall(pattern, reasoning, re.DOTALL)
        
        # Extract just the step content
        reasoning_steps = [step_content.strip() for _, step_content in matches]
        
        return reasoning_steps
    except Exception as e:
        print(f"Error splitting R1 reasoning: {e}")
        return []


def split_reasoning_o3mini(content: str) -> List[str]:
    """Split Claude o3-mini model's reasoning into steps.
    
    Args:
        content: Raw model output text
        
    Returns:
        List of reasoning steps
    """
    try:
        # Extract chain of thought section if present
        if '### Chain of Thought:' in content:
            content = content.split('### Chain of Thought:')[1]
            
        # Extract reasoning part
        if '### Answer:' in content:
            reasoning = content.split('### Answer:')[0]
        else:
            reasoning = content
            
        # Extract steps using regex pattern
        pattern = r"<step\s+(\d+)>\s*(.*?)(?=\n<step\s+\d+>|$)"
        matches = re.findall(pattern, reasoning, re.DOTALL)
        
        # Extract just the step content
        reasoning_steps = [step_content.strip() for _, step_content in matches]
        
        return reasoning_steps
    except Exception as e:
        print(f"Error splitting o3-mini reasoning: {e}")
        return []


def split_reasoning_qwq(content: str) -> List[str]:
    """Split Claude QwQ model's reasoning into steps.
    
    Args:
        content: Raw model output text
        
    Returns:
        List of reasoning steps
    """
    try:
        # Extract reasoning part
        if '### Answer:' in content:
            reasoning = content.split('### Answer:')[0]
        else:
            reasoning = content
            
        # Extract steps using regex pattern
        pattern = r"<step\s+(\d+)>\s*(.*?)(?=\n<step\s+\d+>|$)"
        matches = re.findall(pattern, reasoning, re.DOTALL)
        
        # Extract just the step content
        reasoning_steps = [step_content.strip() for _, step_content in matches]
        
        return reasoning_steps
    except Exception as e:
        print(f"Error splitting QwQ reasoning: {e}")
        return []


def split_reasoning_baichuan(content: str) -> List[str]:
    """Split Baichuan model's reasoning into steps.
    
    Args:
        content: Raw model output text
        
    Returns:
        List of reasoning steps
    """
    try:
        # Extract reasoning part
        if '### Answer:' in content:
            reasoning = content.split('### Answer:')[0]
        else:
            reasoning = content
            
        # Extract steps using regex pattern
        pattern = r"<step\s+(\d+)>\s*(.*?)(?=\n<step\s+\d+>|$)"
        matches = re.findall(pattern, reasoning, re.DOTALL)
        
        # Extract just the step content
        reasoning_steps = [step_content.strip() for _, step_content in matches]
        
        return reasoning_steps
    except Exception as e:
        print(f"Error splitting Baichuan reasoning: {e}")
        return []


def split_reasoning_r1_thinking_process(content: str) -> List[str]:
    """Split DeepSeek-R2 model's reasoning into steps.
    
    Args:
        content: Raw model output text
        
    Returns:
        List of reasoning steps
    """
    try:
        # For R2, reasoning is just separated by newlines
        if '### Answer:' in content:
            reasoning = content.split('### Answer:')[0]
        else:
            reasoning = content
            
        # Clean up double newlines and split by newline
        reasoning = reasoning.replace('\n\n', '\n')
        reasoning_steps = [step.strip() for step in reasoning.split('\n') if step.strip()]
        
        return reasoning_steps
    except Exception as e:
        print(f"Error splitting R2 reasoning: {e}")
        return []


def extract_answer(content: str) -> str:
    """Extract the answer portion from model output.
    
    Args:
        content: Raw model output text
        
    Returns:
        Answer text
    """
    try:
        if '### Answer:' in content:
            answer = content.split('### Answer:')[1].strip()
            return answer
        return ""
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return ""


def get_reasoning_splitter(model_name: str):
    """Get the appropriate reasoning splitter function for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Function that splits reasoning for the specified model
    """
    splitters = {
        'gemini-ft': split_reasoning_gemini,
        'deepseek-r1': split_reasoning_r1,
        'o3-mini': split_reasoning_o3mini,
        'qwq': split_reasoning_qwq,
        'baichuan-m1': split_reasoning_baichuan,
        'deepseek-r1-thinkingprocess': split_reasoning_r1_thinking_process
    }
    
    return splitters.get(model_name, split_reasoning_r1)  # Default to R1 splitter