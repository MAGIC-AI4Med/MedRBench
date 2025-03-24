from web_search import GoogleSearchTool
from openai import OpenAI
import re
import json
import httpx
import time
import numpy as np
from multiprocessing import Pool
import multiprocessing
import os
import random
import argparse
from utils import workflow, workflow_multi_turn, get_reasoning_splitter

# Maximum number of reasoning steps to evaluate
MAX_STEPS = 12
# Number of worker processes for parallel execution
NUM_WORKERS = 8 
# Maximum retry attempts for API calls
MAX_RETRY_ATTEMPTS = 3
# Model to be used for evaluation
EVALUATION_MODEL = "gpt-4o-2024-11-20"

def load_instruction(file_path):
    """Load instruction text from file.
    
    Args:
        file_path: Path to the instruction file
        
    Returns:
        The content of the instruction file as a string
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


def request_correction_from_model(incorrect_output, error_message, retry_count):
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
        model_name=EVALUATION_MODEL, 
        instruction=system_prompt, 
        input_text=input_text
    ).replace('```json', '').replace('```', '').strip()
    
    print(f'Try correct {retry_count}\n before:\n{incorrect_output}\nafter:\n{corrected_completion}')
    
    try:
        output = json.loads(corrected_completion)
        return json.dumps(output)
    except json.JSONDecodeError as e:
        return request_correction_from_model(corrected_completion, str(e), retry_count + 1)


def evaluate_efficiency(current_reasoning_step, previous_reasoning_steps, case_summary, result):
    """Evaluate the efficiency of a single reasoning step.
    
    Args:
        current_reasoning_step: The current reasoning step to evaluate
        previous_reasoning_steps: All previous reasoning steps
        case_summary: Case information summary
        result: Expected result for the case
        
    Returns:
        Efficiency category: 'Citation', 'Repetition', 'Reasoning', or 'Redundancy'
    """
    prompt_template = load_instruction('./instruction/efficiency-en.txt')
    input_text = prompt_template.format(
        current_step=current_reasoning_step,
        previous_steps=previous_reasoning_steps,
        case=case_summary,
        result=result
    )
    system_prompt = 'You are a reliable assistant for the analysis of thought processes.'
    response = workflow(model_name=EVALUATION_MODEL, instruction=system_prompt, input_text=input_text)

    if 'Citation' in response or 'citation' in response:
        return 'Citation'
    elif 'Repetition' in response or 'repetition' in response:
        return 'Repetition'
    elif 'Reasoning' in response or 'reasoning' in response:
        return 'Reasoning'
    elif 'Redundancy' in response or 'redundancy' in response:
        return 'Redundancy'
    else:
        return 'Redundancy'  # Default to redundancy if no clear category detected


def evaluate_factuality(case_info, reasoning_step):
    """Evaluate the factual correctness of a reasoning step.
    
    Args:
        case_info: Case information summary
        reasoning_step: The reasoning step to evaluate
        
    Returns:
        Tuple of (is_correct: bool, judgment_path: list) where judgment_path
        contains the search and evaluation steps taken
    """
    message_history = []
    judgment_path = []
    
    # Extract keywords for search
    keywords_prompt_template = load_instruction('./instruction/keywords-en.txt')
    input_text = keywords_prompt_template.format(case=case_info, reasoning_step=reasoning_step)
    system_prompt = 'You are a professional evaluator of medical knowledge.'
    keywords = workflow(model_name=EVALUATION_MODEL, instruction=system_prompt, input_text=input_text)
    
    # Perform web search
    search_results = GoogleSearchTool(keywords, search_num=3)
    judgment_path.append({
        "judgment": "Search",
        "keywords_to_search": keywords,
        'info': search_results
    })

    # Evaluate factual correctness
    factuality_prompt_template = load_instruction('./instruction/faculity-en.txt')
    input_text = factuality_prompt_template.format(
        case=case_info, 
        reasoning_step=reasoning_step, 
        info=search_results
    )
    message_history.append({"role": "system", "content": system_prompt})
    
    judgment = workflow_multi_turn(
        model_name=EVALUATION_MODEL, 
        input_text=input_text, 
        history_messages=message_history
    )
    message_history.append({"role": "user", "content": input_text})
    message_history.append({"role": "assistant", "content": judgment})

    judgment = judgment.replace('```json', '').replace('```', '').strip()
    judgment = safe_json_parse(judgment)
    
    search_count = 0
    max_searches = 3
    
    # Iterative search if needed
    while judgment['judgment'] == 'Search' and search_count <= max_searches:
        search_keywords = judgment['keywords_to_search']
        search_results = GoogleSearchTool(search_keywords, search_num=3)
        judgment_path.append({
            "judgment": judgment["judgment"],
            "keywords_to_search": judgment["keywords_to_search"],
            'info': search_results
        })
        
        input_text = 'After searching, the following supplementary Known Correct Information has been obtained, please make a judgment again. \n[Known Correct Information]\n' + search_results
        judgment = workflow_multi_turn(
            model_name=EVALUATION_MODEL, 
            input_text=input_text, 
            history_messages=message_history
        )
        message_history.append({"role": "user", "content": input_text})
        message_history.append({"role": "assistant", "content": judgment})
        
        judgment = judgment.replace('```json', '').replace('```', '').strip()
        judgment = safe_json_parse(judgment)
        search_count += 1

    judgment_path.append({
        "judgment": judgment["judgment"],
        "keywords_to_search": judgment["keywords_to_search"]
    })
    
    is_correct = judgment['judgment'] == 'Correct' or judgment['judgment'] == 'correct' or 'Correct' in judgment['judgment']
    return is_correct, judgment_path


def split_ground_truth_reasoning(gt_reasoning):
    """Split ground truth reasoning into individual steps.
    
    Args:
        gt_reasoning: Combined ground truth reasoning text
        
    Returns:
        Formatted string with separated reasoning steps
    """
    prompt_template = load_instruction('./instruction/split-gt-step-en.txt')
    input_text = prompt_template.format(gt_reasoning=gt_reasoning)
    system_prompt = 'You are a reliable thought process organizer.'
    output = workflow(model_name=EVALUATION_MODEL, instruction=system_prompt, input_text=input_text)
    return output


def check_step_hit(ground_truth_step, output_reasoning):
    """Check if a ground truth reasoning step is covered in the output reasoning.
    
    Args:
        ground_truth_step: A single ground truth reasoning step
        output_reasoning: Complete output reasoning text to check against
        
    Returns:
        Boolean indicating whether the step is covered in the output
    """
    prompt_template = load_instruction('./instruction/check-hit-en.txt')
    input_text = prompt_template.format(a_reasoning_step=ground_truth_step, out_reasoning=output_reasoning)
    system_prompt = 'You are a reliable thought process evaluator.'
    output = workflow(model_name=EVALUATION_MODEL, instruction=system_prompt, input_text=input_text)
    return 'yes' in output.lower()


def calculate_efficiency_factuality(evaluated_steps):
    """Calculate efficiency and factuality metrics from evaluated reasoning steps.
    
    Args:
        evaluated_steps: List of evaluated reasoning steps with efficiency and factuality judgments
        
    Returns:
        Tuple of (efficiency_score, factuality_score)
    """
    reasoning_step_count = 0
    correct_step_count = 0
    total_step_count = len(evaluated_steps)
    
    for step in evaluated_steps:
        if step['efficiency'] == 'Reasoning':
            reasoning_step_count += 1
            if step['factulity'] == True:
                correct_step_count += 1
                
    print(f'Total: {total_step_count}, Reasoning: {reasoning_step_count}, Correct: {correct_step_count}')
    
    # Avoid division by zero
    efficiency_score = reasoning_step_count / total_step_count if total_step_count > 0 else 0
    factuality_score = correct_step_count / reasoning_step_count if reasoning_step_count > 0 else 0
    
    return efficiency_score, factuality_score


def calculate_recall(ground_truth_steps):
    """Calculate recall metric based on ground truth step coverage.
    
    Args:
        ground_truth_steps: List of evaluated ground truth steps with hit indicators
        
    Returns:
        Recall score (fraction of ground truth steps covered)
    """
    total_step_count = len(ground_truth_steps)
    hit_count = 0
    
    for step in ground_truth_steps:
        if step['hit'] == True:
            hit_count += 1
            
    print(f'Total: {total_step_count}, Hit: {hit_count}')
    return hit_count / total_step_count if total_step_count > 0 else 0


def evaluate_case(data, save_root, model_name):
    """Evaluate reasoning quality for a specific model's output on a single case.
    
    Args:
        data: Case data including model outputs and ground truth
        save_root: Directory to save evaluation results
        model_name: Name of the model being evaluated
        
    Returns:
        None, results are saved to disk
    """
    print(f'Evaluating case {data["id"]} for model {model_name}')
    error_log_file = f'{model_name}_error.log'
    case_info = data['generate_case']['case_summary']
    reasoning_splitter = get_reasoning_splitter(model_name)
    if model_name == 'deepseek-r1-thinkingprocess':
        all_reasoning_steps = reasoning_splitter(data['results']['thinking_process'])
    else:
        all_reasoning_steps = reasoning_splitter(data['results']['content'])
    combined_reasoning = '\n'.join(all_reasoning_steps)
    reasoning_evaluation_list = []
    
    # Evaluate efficiency and factuality for each reasoning step
    for step_index, reasoning_step in enumerate(all_reasoning_steps):
        if step_index > 0:
            previous_steps = '\n'.join(all_reasoning_steps[:step_index])
        else:
            previous_steps = ''

        # Retry mechanism for efficiency evaluation
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                efficiency_category = evaluate_efficiency(
                    reasoning_step, 
                    previous_steps, 
                    case_info, 
                    data['parsed']['treatment_plan_results']
                )
                break
            except Exception as e:
                with open(error_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"ID: {data['id']}, efficiency_evaluation, Attempt: {attempt + 1}, Error: {str(e)}\n")
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    print(f"Failed to evaluate efficiency after {MAX_RETRY_ATTEMPTS} attempts for ID: {data['id']}")
                    print(str(e))
                    return

        # Evaluate factuality for reasoning steps
        if efficiency_category == 'Reasoning':
            for attempt in range(MAX_RETRY_ATTEMPTS):
                try:
                    is_factual, judgment_path = evaluate_factuality(case_info, reasoning_step)
                    break
                except Exception as e:
                    with open(error_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"ID: {data['id']}, factuality_evaluation, Attempt: {attempt + 1}, Error: {str(e)}\n")
                    if attempt == MAX_RETRY_ATTEMPTS - 1:
                        print(f"Failed to evaluate factuality after {MAX_RETRY_ATTEMPTS} attempts for ID: {data['id']}")
                        print(str(e))
                        return
        else:
            is_factual = None
            judgment_path = []

        reasoning_evaluation_list.append({
            'reasoning_step': reasoning_step,
            'efficiency': efficiency_category,
            'factulity': is_factual,
            'judge_path': judgment_path
        })
        
    data['reasoning_eval'] = reasoning_evaluation_list

    # Evaluate recall against ground truth reasoning
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            gt_text = data['generate_case']["treatment_planning_analysis"] + "\n Treatment plan results:\n" + data['generate_case']["treatment_plan_results"]
            ground_truth_steps_text = split_ground_truth_reasoning(gt_text)
            break
        except Exception as e:
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"ID: {data['id']}, ground_truth_splitting, Attempt: {attempt + 1}, Error: {str(e)}\n")
            if attempt == MAX_RETRY_ATTEMPTS - 1:
                print(f"Failed to split ground truth after {MAX_RETRY_ATTEMPTS} attempts for ID: {data['id']}")
                print(str(e))
                return

    ground_truth_steps = ground_truth_steps_text.replace('\n\n', '\n').split('\n')
    ground_truth_steps = [step.strip() for step in ground_truth_steps if step.strip() != '']
    
    ground_truth_evaluation_list = []
    for ground_truth_step in ground_truth_steps:
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                is_hit = check_step_hit(ground_truth_step, combined_reasoning)
                break
            except Exception as e:
                with open(error_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"ID: {data['id']}, hit_checking, Attempt: {attempt + 1}, Error: {str(e)}\n")
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    print(f"Failed to check hit after {MAX_RETRY_ATTEMPTS} attempts for ID: {data['id']}")
                    print(str(e))
                    return
                    
        ground_truth_evaluation_list.append({'reasoning_step': ground_truth_step, 'hit': is_hit})
        
    data['gt_reasoning_eval'] = ground_truth_evaluation_list

    # Calculate overall metrics
    efficiency_score, factuality_score = calculate_efficiency_factuality(data['reasoning_eval'])
    recall_score = calculate_recall(data['gt_reasoning_eval'])
    
    data['efficiency'] = efficiency_score
    data['factulity'] = factuality_score
    data['recall'] = recall_score

    # Save evaluation results
    with open(os.path.join(save_root, f'{data["id"]}.json'), 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def worker(task_queue):
    """Worker process function to process evaluation tasks from queue.
    
    Args:
        task_queue: Queue containing evaluation tasks
        
    Returns:
        None
    """
    while not task_queue.empty():
        try:
            data, save_root, model_name = task_queue.get()
            evaluate_case(data, save_root, model_name)
        except Exception as e:
            print(f"Worker error: {e}")


def main(model_name, patient_case_filepath, model_output_filepath, output_directory, use_parallel=True):
    """Main function to orchestrate the evaluation process.
    
    Args:
        model_name: Name of the model to evaluate
        patient_case_filepath: Path to file containing patient cases
        model_output_filepath: Path to file containing model outputs
        output_directory: Directory to save evaluation results
        use_parallel: Whether to use parallel processing
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Load patient cases and model outputs
    with open(patient_case_filepath, 'r', encoding='utf-8') as f:
        patient_cases = json.load(f)
    
    with open(model_output_filepath, 'r', encoding='utf-8') as f:
        model_outputs = json.load(f)
        

    # Filter already processed data
    cases_to_evaluate = []
    
    completed_cases = os.listdir(output_directory)
    completed_case_ids = [name.split('.')[0] for name in completed_cases]
    
    for case_id in patient_cases.keys():
        if case_id not in completed_case_ids:
            case_data = patient_cases[case_id]
            case_data['id'] = case_id
            case_data['results'] = model_outputs[case_id][model_name]
            cases_to_evaluate.append(case_data)    
    
    print(f'Total cases to evaluate: {len(cases_to_evaluate)}')

    if use_parallel and len(cases_to_evaluate) > 0:
        # Create multiprocessing task queue
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        
        for case_data in cases_to_evaluate:
            task_queue.put((case_data, output_directory, model_name))

        # Start worker processes
        processes = []
        worker_count = min(NUM_WORKERS, len(cases_to_evaluate))
        for _ in range(worker_count):
            p = multiprocessing.Process(target=worker, args=(task_queue,))
            p.start()
            processes.append(p)

        # Wait for completion
        for p in processes:
            p.join()
    else:
        for case_data in cases_to_evaluate:
            evaluate_case(case_data, output_directory, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model accuracy on treatment planning tasks')
    parser.add_argument('--model', type=str, required=True, 
                      choices=['qwq', 'o3-mini', 'gemini2-ft', 'deepseek-r1', 'baichuan-m1', 'deepseek-r1-thinkingprocess'],
                      help='Model to evaluate')
    parser.add_argument('--sequential', action='store_true', 
                      help='Run sequentially instead of using parallel processing')
    
    args = parser.parse_args()
    
    # Define input and output file paths
    model_output_filepath = '../../../data/InferenceResults/treatment_planning.json'
    patient_case_filepath = '../../../data/MedRBench/treatment_496_cases_with_rare_disease_165.json'
    output_directory = f'./reasoning_results/{args.model}'
    
    # Run main evaluation process
    main(
        args.model, 
        patient_case_filepath, 
        model_output_filepath, 
        output_directory, 
        not args.sequential
    )