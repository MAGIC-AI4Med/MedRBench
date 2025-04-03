from utils import workflow, load_instruction, safe_json_parse

def parse_info_requirements(info_required_text):
    """Parse information requirements into structured format.
    
    Args:
        info_required_text: String containing information requirements
        
    Returns:
        List of dictionaries containing structured information requirements
    """
    split_prompt = load_instruction('./instructions/parse_info_requirements.txt').format(info_required=info_required_text)
    formatted_info = workflow(input_text=split_prompt, Instruction='You are a medical text organizer.')
    
    formatted_info = formatted_info.replace('```json', '').replace('```', '').strip()
    structured_info = safe_json_parse(formatted_info)
    
    # Ensure result is a list even if only one item is returned
    if isinstance(structured_info, dict):
        structured_info = [structured_info]
    
    # Normalize keys if needed
    for i in range(len(structured_info)):
        if 'info_requried' in structured_info[i]:
            structured_info[i] = {
                ('info_required' if k == 'info_requried' else k): v for k, v in structured_info[i].items()
            }
    
    return structured_info

def is_requirement_matched(requirement, reference_text):
    """Check if an information requirement is covered in the reference text.
    
    Args:
        requirement: Dictionary containing structured information requirement
        reference_text: String containing reference text to check against
        
    Returns:
        Boolean indicating whether the requirement is covered in the reference
    """
    if 'type' not in requirement.keys():
        return False
    
    requirement_description = 'Category of Test: {}, Name of Test Item: {}, Purpose of the Test or Information Desired: {}'.format(
        requirement['type'], 
        requirement['test_name'], 
        requirement['info_required']
    )
    
    prompt = load_instruction('./instructions/is_requirement_matched.txt').format(
        a_info_step=requirement_description, 
        gt_info=reference_text
    )
    
    response = workflow(
        input_text=prompt,
        Instruction='You are a medical assistant examination specialist and a medical text comparison assistant.'
    )
    
    return any(keyword in response for keyword in ['是', 'Yes', 'yes', 'YES'])

def calculate_match_rate(requirements_with_match_status):
    """Calculate percentage of requirements that were matched.
    
    Args:
        requirements_with_match_status: List of dictionaries containing requirements with 'hit' field
        
    Returns:
        Percentage of requirements marked as matched
    """
    total_count = len(requirements_with_match_status)
    matched_count = sum(1 for req in requirements_with_match_status if req['hit'] == True)
    
    return matched_count / total_count if total_count > 0 else 0

def eval_dynamic_asking_info_precision_recall(pred_requirements, gt_requirements, max_retries=3):
    """Evaluate precision and recall of predicted information requirements.
    
    Args:
        pred_requirements: String containing predicted information requirements
        gt_requirements: String containing ground truth information requirements
        max_retries: Maximum number of retry attempts for error handling
        
    Returns:
        Dictionary containing precision, recall, and structured requirements with match status
    """
    # Parse predicted information requirements
    for attempt in range(max_retries):
        try:
            parsed_pred_requirements = parse_info_requirements(pred_requirements)
            if len(parsed_pred_requirements) == 1 and 'type' not in parsed_pred_requirements[0].keys():
                # Model indicates no additional information needed
                return {
                    'precision': 0.0,
                    'recall': 0.0,
                    'infer_info_split': [],
                    'gt_info_split': []
                }
            break
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": f"Failed to parse prediction requirements: {str(e)}"}
    
    # Parse ground truth information requirements
    for attempt in range(max_retries):
        try:
            parsed_gt_requirements = parse_info_requirements(gt_requirements)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": f"Failed to parse ground truth requirements: {str(e)}"}
    
    # Evaluate precision - check if predicted requirements match ground truth
    for i, requirement in enumerate(parsed_pred_requirements):
        for attempt in range(max_retries):
            try:
                is_matched = is_requirement_matched(requirement, gt_requirements)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    is_matched = False
        
        parsed_pred_requirements[i]['hit'] = is_matched
    
    precision = calculate_match_rate(parsed_pred_requirements)
    
    # Evaluate recall - check if ground truth requirements are covered in prediction
    for i, requirement in enumerate(parsed_gt_requirements):
        for attempt in range(max_retries):
            try:
                is_matched = is_requirement_matched(requirement, pred_requirements)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    is_matched = False
        
        parsed_gt_requirements[i]['hit'] = is_matched
    
    recall = calculate_match_rate(parsed_gt_requirements)
    
    return {
        'precision': precision,
        'recall': recall,
        'infer_info_split': parsed_pred_requirements,
        'gt_info_split': parsed_gt_requirements
    }