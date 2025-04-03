import re
from typing import List, Optional

def get_reasoning_content(content: str) -> str:
    content = content.replace('```', '').strip()
    if '### Reasoning:' in content:
        content = content.split('### Reasoning:')[1].strip()
    elif '**Reasoning:**' in reasoning_text:
        reasoning_text = reasoning_text.split('**Reasoning:**')[1].strip()
    elif '### Chain of Thought:' in content:
        content = content.split('### Chain of Thought:')[1].strip()
                
    if '### Answer:' in content:
        content = content.split('### Answer:')[0].strip()
        
    return content.strip()

def split_reasoning(content: str, max_steps=10) -> List[str]:
    try:
        reasoning = get_reasoning_content(content)
        
        # Extract steps using regex pattern <step 1> <step 2>
        pattern = r"<step\s+(\d+)>\s*(.*?)(?=\n<step\s+\d+>|$)"
        matches = re.findall(pattern, reasoning, re.DOTALL)
        reasoning_steps = [step_content.strip() for _, step_content in matches]
        
        if len(reasoning_steps) == 0:
            # If no steps found, try splitting by newlines
            if '\n\n' in reasoning:
                reasoning_steps = reasoning.split('\n\n')
            else:
                reasoning_steps = reasoning.split('\n')
        return reasoning_steps[:max_steps]  # Limit to max_steps if specified
    except Exception as e:
        print(f"Error extracting reasoning content: {e}")
        return []
    
def extract_ancillary_tests(case_summary):
    """
    Extract ancillary tests from case summary and split into separate sections.
    
    Args:
        case_summary (str): The full case summary text    
    Returns:
        tuple: A tuple containing the case summary without ancillary tests and the ancillary tests section.
    """
    if "Ancillary Tests" not in case_summary:
        return False
    
    case_summary_paragraphs = case_summary.strip().split('\n')
    
    for idx, paragraph in enumerate(case_summary_paragraphs):
        if "Ancillary Tests" in paragraph:
            ancillary_test = "\n".join(case_summary_paragraphs[idx:])
            case_summary_without_ancillary_test = "\n".join(case_summary_paragraphs[:idx])
            
            return case_summary_without_ancillary_test, ancillary_test
    
    return case_summary, None