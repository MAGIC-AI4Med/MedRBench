import os
import json
import logging
import argparse
import multiprocessing
from multiprocessing import Manager

from utils import split_reasoning
from metrics.reasoning_eval import (
    eval_reasoning_efficiency_factuality,
    eval_reasoning_completeness
)

# Configuration constants
NUM_WORKERS = 8  # Number of worker processes for parallel execution
MAX_RETRY_ATTEMPTS = 3  # Maximum retry attempts for API calls
EVALUATION_MODEL = "gpt-4o-2024-11-20"  # Model to be used for evaluation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def evaluate_case(data, save_root, model_name):
    """Evaluate reasoning quality for a specific model's output on a single case."""
    logger.info(f'Evaluating case {data["id"]} for model {model_name}')
    error_log_file = f'{model_name}_error.log'
    
    try:
        case_info = data['generate_case']['case_summary']
        gt_answer = data['generate_case']['final_diagnosis']
        gt_reasoning = data['generate_case']["differential_diagnosis"] + "\n Final diagnosis:\n" + data['generate_case']["final_diagnosis"]
               
        # Extract reasoning steps based on model type
        if model_name == 'deepseek-r1-thinkingprocess':
            reasoning_steps = split_reasoning(data['results']['thinking_process'])
        else:
            reasoning_steps = split_reasoning(data['results']['content'])
            
        # Combine all steps for recall evaluation
        combined_reasoning = '\n'.join(reasoning_steps)
        
        # Evaluate efficiency and factuality
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                efficiency_factuality_results = eval_reasoning_efficiency_factuality(
                    case_info=case_info,
                    pred_reasoning_steps_list=reasoning_steps,
                    gt_answer=gt_answer,
                    is_treatment=False,
                    evaluation_model=EVALUATION_MODEL
                )
                break
            except Exception as e:
                with open(error_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"ID: {data['id']}, efficiency_factuality_evaluation, Attempt: {attempt + 1}, Error: {str(e)}\n")
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    logger.error(f"Failed to evaluate efficiency/factuality after {MAX_RETRY_ATTEMPTS} attempts for ID: {data['id']}")
                    logger.error(str(e))
                    return
        
        # Evaluate completeness
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                completeness_results = eval_reasoning_completeness(
                    gt_reasoning=gt_reasoning,
                    pred_reasoning_steps_string=combined_reasoning,
                    evaluation_model=EVALUATION_MODEL
                )
                break
            except Exception as e:
                with open(error_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"ID: {data['id']}, completeness_evaluation, Attempt: {attempt + 1}, Error: {str(e)}\n")
                if attempt == MAX_RETRY_ATTEMPTS - 1:
                    logger.error(f"Failed to evaluate completeness after {MAX_RETRY_ATTEMPTS} attempts for ID: {data['id']}")
                    logger.error(str(e))
                    return
        
        # Store evaluation results
        data['reasoning_eval'] = efficiency_factuality_results['evaluated_steps']
        data['gt_reasoning_eval'] = completeness_results['ground_truth_steps']
        data['efficiency'] = efficiency_factuality_results['efficiency_score']
        data['factulity'] = efficiency_factuality_results['factuality_score']
        data['recall'] = completeness_results['recall_score']

        # Save evaluation results
        with open(os.path.join(save_root, f'{data["id"]}.json'), 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        logger.error(f"Error evaluating case {data['id']}: {str(e)}")
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(f"ID: {data['id']}, general_error: {str(e)}\n")


def worker(task_queue):
    """Worker process function to process evaluation tasks from queue."""
    while not task_queue.empty():
        try:
            data, save_root, model_name = task_queue.get()
            evaluate_case(data, save_root, model_name)
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")


def main(model_name, patient_case_filepath, model_output_filepath, output_directory, use_parallel=True):
    """Main function to orchestrate the evaluation process."""
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
        if case_id not in completed_case_ids and case_id in model_outputs and model_name in model_outputs[case_id]:
            case_data = patient_cases[case_id].copy()  # Create a copy to avoid modifying the original
            case_data['id'] = case_id
            case_data['results'] = model_outputs[case_id][model_name]
            cases_to_evaluate.append(case_data)    
    
    logger.info(f'Total cases to evaluate: {len(cases_to_evaluate)}')

    if use_parallel and len(cases_to_evaluate) > 0:
        # Create multiprocessing task queue
        manager = Manager()
        task_queue = manager.Queue()
        
        for case_data in cases_to_evaluate:
            task_queue.put((case_data, output_directory, model_name))

        # Start worker processes
        processes = []
        worker_count = min(NUM_WORKERS, len(cases_to_evaluate))
        logger.info(f"Starting {worker_count} worker processes")
        
        for _ in range(worker_count):
            p = multiprocessing.Process(target=worker, args=(task_queue,))
            p.start()
            processes.append(p)

        # Wait for completion
        for p in processes:
            p.join()
    else:
        logger.info("Processing cases sequentially")
        for case_data in cases_to_evaluate:
            evaluate_case(case_data, output_directory, model_name)
            
    logger.info(f"Evaluation completed for model {model_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model reasoning on treatment planning tasks')
    parser.add_argument('--model', type=str, required=True, 
                      choices=['qwq', 'o3-mini', 'gemini2-ft', 'deepseek-r1', 'baichuan-m1', 'deepseek-r1-thinkingprocess'],
                      help='Model to evaluate')
    parser.add_argument('--sequential', action='store_true', 
                      help='Run sequentially instead of using parallel processing')
    parser.add_argument('--output-dir', type=str, default='./reasoning_results',
                      help='Base directory for evaluation results')
    parser.add_argument('--patient-cases', type=str,
                      default='../../../data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json',
                      help='Path to patient cases file')
    parser.add_argument('--model-outputs', type=str,
                      default='../../../data/InferenceResults/oracle_diagnosis.json',
                      help='Path to model outputs file')
    
    args = parser.parse_args()
    
    # Define input and output file paths
    model_output_filepath = args.model_outputs
    patient_case_filepath = args.patient_cases
    output_directory = f'{args.output_dir}/{args.model}'
    
    # Run main evaluation process
    main(
        args.model, 
        patient_case_filepath, 
        model_output_filepath, 
        output_directory, 
        not args.sequential
    )