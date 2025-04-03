from utils import workflow, load_instruction
from web_search import BingSearchTool

def eval_accuracy_with_websearch(pred_outcome_answer, gt_outcome_answer, case_info, evaluation_model = "gpt-4o-2024-11-20"):
    """Evaluate the accuracy of a treatment prediction against ground truth wtih websearch.
    
    Args:
        pred_outcome_answer: Model's predicted answewr
        gt_outcome_answer: Ground truth answer
        case_info: Additional case information for context
        
    Returns:
        Tuple containing (keywords, search_results, is_correct_boolean)
    """

    # Generate keywords for information retrieval
    keywords_prompt_template = load_instruction('./instruction/treatment_plan_extract_keywords.txt')
    keywords_prompt = keywords_prompt_template.format(info=case_info)
    system_prompt = 'You are a professional evaluator of medical knowledge.'
    keywords = workflow(model_name=evaluation_model, instruction=system_prompt, input_text=keywords_prompt)
    
    # Retrieve relevant medical information
    search_results = BingSearchTool(keywords, return_num=3)
   
    # Evaluate accuracy with retrieved information
    evaluation_template = load_instruction('./instruction/acc_treatment_plan.txt')
    evaluation_prompt = evaluation_template.format(
        pred_treatment=pred_outcome_answer, 
        gt_treatment=gt_outcome_answer, 
        additional_info=search_results
    )
    system_prompt = 'You are a professional medical diagnosis evaluation system.'
    evaluation_result = workflow(model_name=evaluation_model, instruction=system_prompt, input_text=evaluation_prompt)
    
    is_correct = 'correct' in evaluation_result.lower()
    return keywords, search_results, is_correct


def eval_accuracy(pred_outcome_answer, gt_outcome_answer, evaluation_model = "gpt-4o-2024-11-20"):
    """Evaluate the accuracy of a diagnosis prediction against ground truth.
    
    Args:
        pred_outcome_answer: Model's predicted diagnosis
        gt_outcome_answer: Ground truth diagnosis
        
    Returns:
        is_correct_boolean
    """
    evaluation_template = load_instruction('./instructions/acc_diagnose.txt')
    evaluation_prompt = evaluation_template.format(
        pred_diagnose=pred_outcome_answer, 
        gt_diagnose=gt_outcome_answer, 
    )
    system_prompt = 'You are a professional medical diagnosis evaluation system.'
    evaluation_result = workflow(model_name=evaluation_model, instruction=system_prompt, input_text=evaluation_prompt)
    
    is_correct = 'correct' in evaluation_result.lower()
    return is_correct


