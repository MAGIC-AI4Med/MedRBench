from web_search import BingSearchTool
from utils import workflow, workflow_multi_turn, load_instruction, safe_json_parse

# -------------- Evaluate one step efficiency -------------- # 

def evaluate_efficiency(current_reasoning_step, previous_reasoning_steps, case_summary, result, evaluation_model = 'gpt-4o-2024-11-20'):
    """Evaluate the efficiency of a single reasoning step.
    
    Args:
        current_reasoning_step: The current reasoning step to evaluate
        previous_reasoning_steps: All previous reasoning steps
        case_summary: Case information summary
        result: Expected result for the case
        
    Returns:
        Efficiency category: 'Citation', 'Repetition', 'Reasoning', or 'Redundancy'
    """
    prompt_template = load_instruction('./instructions/reasoning_efficiency.txt')
    input_text = prompt_template.format(
        current_step=current_reasoning_step,
        previous_steps=previous_reasoning_steps,
        case=case_summary,
        result=result
    )
    system_prompt = 'You are a reliable assistant for the analysis of thought processes.'
    response = workflow(model_name=evaluation_model, instruction=system_prompt, input_text=input_text)

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


# -------------- Evaluate one step factulity -------------- # 
def evaluate_factuality(case_info, reasoning_step, evaluation_model='gpt-4o-2024-11-20', is_treatment=False):
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
    if is_treatment:
        keywords_prompt_template = load_instruction('./instructions/treatment_plan_extract_keywords.txt')
    else:
        keywords_prompt_template = load_instruction('./instructions/extract_keywords.txt')
    input_text = keywords_prompt_template.format(case=case_info, reasoning_step=reasoning_step)
    system_prompt = 'You are a professional evaluator of medical knowledge.'
    keywords = workflow(model_name=evaluation_model, instruction=system_prompt, input_text=input_text)
    
    # Perform web search
    search_results = BingSearchTool(keywords, search_num=3)
    judgment_path.append({
        "judgment": "Search",
        "keywords_to_search": keywords,
        'info': search_results
    })

    # Evaluate factual correctness
    factuality_prompt_template = load_instruction('./instructions/reasoning_factuality.txt')
    input_text = factuality_prompt_template.format(
        case=case_info, 
        reasoning_step=reasoning_step, 
        info=search_results
    )
    message_history.append({"role": "system", "content": system_prompt})
    
    judgment = workflow_multi_turn(
        model_name=evaluation_model, 
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
        search_results = BingSearchTool(search_keywords, search_num=3)
        judgment_path.append({
            "judgment": judgment["judgment"],
            "keywords_to_search": judgment["keywords_to_search"],
            'info': search_results
        })
        
        input_text = 'After searching, the following supplementary Known Correct Information has been obtained, please make a judgment again. \n[Known Correct Information]\n' + search_results
        judgment = workflow_multi_turn(
            model_name=evaluation_model, 
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

# -------------- Evaluate Recall -------------- # 
def split_ground_truth_reasoning(gt_reasoning, evaluation_model = 'gpt-4o-2024-11-20'):
    """Split ground truth reasoning into individual steps.
    
    Args:
        gt_reasoning: Combined ground truth reasoning text
        
    Returns:
        Formatted string with separated reasoning steps
    """
    prompt_template = load_instruction('./instructions/reasoning_split_gt_steps.txt')
    input_text = prompt_template.format(gt_reasoning=gt_reasoning)
    system_prompt = 'You are a reliable thought process organizer.'
    output = workflow(model_name=evaluation_model, instruction=system_prompt, input_text=input_text)
    return output


def check_step_hit(ground_truth_step, output_reasoning, evaluation_model = 'gpt-4o-2024-11-20'):
    """Check if a ground truth reasoning step is covered in the output reasoning.
    
    Args:
        ground_truth_step: A single ground truth reasoning step
        output_reasoning: Complete output reasoning text to check against
        
    Returns:
        Boolean indicating whether the step is covered in the output
    """
    prompt_template = load_instruction('./instructions/reasoning_check_hit.txt')
    input_text = prompt_template.format(a_reasoning_step=ground_truth_step, out_reasoning=output_reasoning)
    system_prompt = 'You are a reliable thought process evaluator.'
    output = workflow(model_name=evaluation_model, instruction=system_prompt, input_text=input_text)
    return 'yes' in output.lower()

# -------------- Evaluate one case -------------- # 
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

def eval_reasoning_efficiency_factuality(case_info, pred_reasoning_steps_list, gt_answer, is_treatment, evaluation_model = 'gpt-4o-2024-11-20'):
    """Evaluate efficiency and factuality of reasoning steps.
    
    Args:
        case_info: Case information summary
        pred_reasoning_steps_list: List of predicted reasoning steps to evaluate
        gt_answer: Ground truth answer for the case
        evaluation_model: Model to use for evaluation
        
    Returns:
        Dictionary containing efficiency score, factuality score, and detailed evaluations
    """
    evaluated_steps = []
    
    # Evaluate each reasoning step
    for i, reasoning_step in enumerate(pred_reasoning_steps_list):
        # Prepare previous steps text for efficiency evaluation
        if i > 0:
            previous_steps = '\n'.join(pred_reasoning_steps_list[:i])
        else:
            previous_steps = ''
            
        # Evaluate efficiency
        efficiency_category = evaluate_efficiency(
            current_reasoning_step=reasoning_step, 
            previous_reasoning_steps=previous_steps, 
            case_summary=case_info, 
            result=gt_answer,
            evaluation_model=evaluation_model
        )
        
        # Evaluate factuality only for reasoning steps
        if efficiency_category == 'Reasoning':
            is_factual, judgment_path = evaluate_factuality(
                case_info=case_info, 
                reasoning_step=reasoning_step,
                evaluation_model=evaluation_model,
                is_treatment=is_treatment
            )
        else:
            is_factual = None
            judgment_path = []
            
        # Add evaluation results for this step
        evaluated_steps.append({
            'step': reasoning_step,
            'efficiency': efficiency_category,
            'factulity': is_factual,
            'judgment_path': judgment_path
        })
    
    # Calculate overall efficiency and factuality scores
    efficiency_score, factuality_score = calculate_efficiency_factuality(evaluated_steps)
    
    return {
        'efficiency_score': efficiency_score,
        'factuality_score': factuality_score,
        'evaluated_steps': evaluated_steps
    }


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


def eval_reasoning_completeness(gt_reasoning, pred_reasoning_steps_string, evaluation_model = 'gpt-4o-2024-11-20'):
    """Evaluate completeness of model reasoning against ground truth reasoning.
    
    Args:
        gt_reasoning: Ground truth reasoning text
        pred_reasoning_steps_string: Combined string of predicted reasoning steps
        evaluation_model: Model to use for evaluation
        
    Returns:
        Dictionary containing recall score and detailed evaluation of ground truth steps
    """
    # Split ground truth reasoning into individual steps
    ground_truth_steps_text = split_ground_truth_reasoning(
        gt_reasoning=gt_reasoning,
        evaluation_model=evaluation_model
    )
    
    # Process the ground truth steps text into a list
    ground_truth_steps = ground_truth_steps_text.replace('\n\n', '\n').split('\n')
    ground_truth_steps = [step.strip() for step in ground_truth_steps if step.strip() != '']
    
    # Evaluate each ground truth step for coverage
    ground_truth_evaluation = []
    for ground_truth_step in ground_truth_steps:
        is_hit = check_step_hit(
            ground_truth_step=ground_truth_step, 
            output_reasoning=pred_reasoning_steps_string,
            evaluation_model=evaluation_model
        )
        
        ground_truth_evaluation.append({
            'step': ground_truth_step,
            'hit': is_hit
        })
    
    # Calculate overall recall score
    recall_score = calculate_recall(ground_truth_evaluation)
    
    return {
        'recall_score': recall_score,
        'ground_truth_steps': ground_truth_evaluation
    }
