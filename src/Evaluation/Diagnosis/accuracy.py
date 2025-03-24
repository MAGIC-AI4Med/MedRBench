import os
import json
import argparse
import multiprocessing
from utils import workflow

# Configuration constants
NUM_WORKERS = 4  # Number of parallel worker processes
EVALUATION_MODEL = "gpt-4o-2024-11-20"  # Language model used for evaluation

def load_instruction(file_path):
    """Load instruction text from file.
    
    Args:
        file_path: Path to the instruction template file
        
    Returns:
        String containing the instruction template
    """
    with open(file_path, 'r', encoding='utf-8') as fp:
        return fp.read()

def evaluate_accuracy(prediction, ground_truth):
    """Evaluate the accuracy of a diagnosis prediction against ground truth.
    
    Args:
        prediction: Model's predicted diagnosis
        ground_truth: Ground truth diagnosis
        case_info: Additional case information for context
        
    Returns:
        Tuple containing (keywords, search_results, is_correct_boolean)
    """
    # Extract answer content if needed
    if '### Answer' in prediction:
        prediction = prediction.split('### Answer')[-1].replace('\n', '').replace(':', '')
   
    # Evaluate accuracy with retrieved information
    evaluation_template = load_instruction('./instruction/acc.txt')
    evaluation_prompt = evaluation_template.format(
        pred_diag=prediction, 
        gt_diag=ground_truth, 
    )
    system_prompt = 'You are a professional medical diagnosis evaluation system.'
    evaluation_result = workflow(model_name=EVALUATION_MODEL, instruction=system_prompt, input_text=evaluation_prompt)
    
    is_correct = 'correct' in evaluation_result.lower()
    return is_correct

def evaluate_case(case_data, output_directory, model_name):
    """Evaluate a single case and save results.
    
    Args:
        case_data: Dictionary containing case information and model outputs
        output_directory: Directory to save evaluation results
        model_name: Name of the model being evaluated
        
    Returns:
        None (results are saved to disk)
    """
    print(f'Evaluating case {case_data["id"]} for model {model_name}')

    try:
        # Get ground truth and model prediction
        ground_truth = case_data['generate_case']['diagnosis_results']
        model_prediction = case_data['results']['content']
        
        # Evaluate accuracy
        is_accurate = evaluate_accuracy(
            model_prediction, 
            ground_truth, 
        )
   
        case_data['accuracy'] = is_accurate

        # Save results to file
        output_file = os.path.join(output_directory, f'{case_data["id"]}.json')
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(case_data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        print(f'Error processing case {case_data["id"]}: {str(e)}')
    

def worker_process(task_queue):
    """Process evaluation tasks from a queue.
    
    Args:
        task_queue: Queue containing evaluation tasks
        
    Returns:
        None
    """
    while not task_queue.empty():
        try:
            case_data, output_directory, model_name = task_queue.get()
            evaluate_case(case_data, output_directory, model_name)
        except Exception as e:
            print(f"Worker error: {e}")

def main(model_name, patient_case_filepath, model_output_filepath, output_directory, use_parallel=True):
    """Orchestrate the evaluation process for a specific model.
    
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
    
    # Identify cases that need to be processed
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
        # Parallel processing approach
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        
        # Add all tasks to queue
        for case_data in cases_to_evaluate:
            task_queue.put((case_data, output_directory, model_name))

        # Create and start worker processes
        processes = []
        worker_count = min(NUM_WORKERS, len(cases_to_evaluate))
        for _ in range(worker_count):
            process = multiprocessing.Process(target=worker_process, args=(task_queue,))
            process.start()
            processes.append(process)

        # Wait for all processes to complete
        for process in processes:
            process.join()
    else:
        # Sequential processing approach
        for case_data in cases_to_evaluate:
            evaluate_case(case_data, output_directory, model_name)


if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Evaluate model accuracy on diagnose tasks')
    parser.add_argument('--model', type=str, required=True, 
                      choices=['qwq', 'o3-mini', 'gemini2-ft', 'deepseek-r1', 'baichuan-m1'],
                      help='Model to evaluate')
    parser.add_argument('--sequential', action='store_true', 
                      help='Run sequentially instead of using parallel processing')
    
    args = parser.parse_args()
    
    # Define input and output file paths
    model_output_filepath = '../../../data/InferenceResults/oracle_diagnosis.json'
    patient_case_filepath = '../../../data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json'
    output_directory = f'./acc_results/{args.model}'
    
    # Run main evaluation process
    main(
        args.model, 
        patient_case_filepath, 
        model_output_filepath, 
        output_directory, 
        not args.sequential
    )