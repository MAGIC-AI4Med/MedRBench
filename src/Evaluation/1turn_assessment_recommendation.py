import os
import json
import logging
import argparse
import multiprocessing
from multiprocessing import Manager

from utils import split_reasoning, extract_ancillary_tests
from metrics.assessment_recommendation_eval import eval_dynamic_asking_info_precision_recall

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
    case_id = data["id"]
    logger.info(f'Evaluating case {case_id} for model {model_name}')
    
    try:
        # Extract case information
        case_info = data['generate_case']['case_summary']
        case_info_without_ancillary_test, ancillary_test = extract_ancillary_tests(case_info)
        gt_reasoning = data['generate_case']["differential_diagnosis"] + "\nFinal diagnosis:\n" + data['generate_case']["final_diagnosis"]
        
        # Get model outputs
        model_output = data['results']['messages'][2]['content']['answer']
        pred_info_required = model_output.split('### Additional Information Required:')[-1] if '### Additional Information Required:' in model_output else ""
        gt_info_required = ancillary_test if ancillary_test else ""
        
        # Evaluate with retries
        eval_results = None
        last_error = None
        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                eval_results = eval_dynamic_asking_info_precision_recall(
                    pred_info_required,
                    gt_info_required
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for case {case_id}: {str(e)}")
                continue
        
        if not eval_results:
            logger.error(f"Failed to evaluate case {case_id} after {MAX_RETRY_ATTEMPTS} attempts: {str(last_error)}")
            return
        
        if 'error' in eval_results:
            logger.error(f"Evaluation error for case {case_id}: {eval_results['error']}")
            return
            
        # Store results
        data['evaluation'] = {
            'precision': eval_results['precision'],
            'recall': eval_results['recall'],
            'infer_info_split': eval_results['infer_info_split'],
            'gt_info_split': eval_results['gt_info_split']
        }
        
        # Save results
        output_path = os.path.join(save_root, f'{case_id}.json')
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        logger.error(f"Unexpected error evaluating case {case_id}: {str(e)}", exc_info=True)


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
                      default='../../../data/InferenceResults/1turn_assessment_recommendation+final_diagnosis.json',
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