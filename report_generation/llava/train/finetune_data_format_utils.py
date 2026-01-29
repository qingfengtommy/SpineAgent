import re
import json
import os


def has_all_four_choices(question_text):
    """Check if question contains all four choices (a), (b), (c), (d)."""
    required_choices = ['(a)', '(b)', '(c)', '(d)']
    return all(choice in question_text for choice in required_choices)


def parse_qa_sections(qa_str):
    """Parse QA string into a dict of sections, each with list of (q, a) tuples."""
    sections = {}
    for section in ['free_response', 'description', 'multiple_choice']:
        pattern = f"<{section}>(.*?)</{section}>"
        match = re.search(pattern, qa_str, re.DOTALL)
        if match:
            content = match.group(1)
            
            if section == 'multiple_choice':
                # For multiple choice, we need to include choices in the question
                # First try to find pattern with separate <choices> tag
                mc_pattern = r"<q>(.*?)</q><choices>(.*?)</choices><a>(.*?)</a>"
                mc_matches = re.findall(mc_pattern, content, re.DOTALL)
                
                if mc_matches:
                    # Format: <q>question</q><choices>choices</choices><a>answer</a>
                    valid_mc_pairs = []
                    for q, choices, a in mc_matches:
                        full_question = f"{q.strip()} {choices.strip()}"
                        if has_all_four_choices(full_question):
                            valid_mc_pairs.append((full_question, a.strip()))
                    sections[section] = valid_mc_pairs
                else:
                    # Fallback: try to find choices pattern within <q> tag or use standard pattern
                    qas = re.findall(r"<q>(.*?)</q><a>(.*?)</a>", content, re.DOTALL)
                    processed_qas = []
                    for q, a in qas:
                        question = q.strip()
                        # Check if question already contains choices pattern like (a) (b) (c) (d)
                        if re.search(r'\([a-z]\)', question):
                            # Only include if all four choices are present
                            if has_all_four_choices(question):
                                processed_qas.append((question, a.strip()))
                        else:
                            # Look for choices after the question in the same section
                            # This is a fallback - you may need to adjust based on actual input format
                            # Only include if all four choices are present
                            if has_all_four_choices(question):
                                processed_qas.append((question, a.strip()))
                    sections[section] = processed_qas
            else:
                # For free_response and description, use the original pattern
                qas = re.findall(r"<q>(.*?)</q><a>(.*?)</a>", content, re.DOTALL)
                sections[section] = [(q.strip(), a.strip()) for q, a in qas]
        else:
            sections[section] = []
    return sections

def transform_qa_conversation_to_targeted_format(conversation_data):
    """
    Transform the input QA conversation format to the targeted QA pair format.
    
    Input format: conversation_data (string with <free_response>, <description>, <multiple_choice> sections)
    Output format: list of dicts with 'from', 'value', 'type' keys
    
    Example input:
    "<free_response><q>Question1</q><a>Answer1</a><q>Question2</q><a>Answer2</a></free_response>"
    
    Example output:
    [
        {"from": "human", "value": "<image>\nQuestion1<long_answer>", "type": "free_response"},
        {"from": "gpt", "value": "Answer1"},
        {"from": "human", "value": "Question2<long_answer>", "type": "free_response"},
        {"from": "gpt", "value": "Answer2"}
    ]
    """
    targeted_qa_pairs = []
    # Parse the conversation data into sections
    sections = parse_qa_sections(conversation_data)
    
    for section_name, qa_list in sections.items():
        current_section_qa_pairs = []
        for i, (question, answer) in enumerate(qa_list):
            # Add human question
            if i == 0 and section_name == "free_response":
                human_value = f"<image>\n{question}<long_answer>"
            elif i == 0 and section_name == "description":
                human_value = f"<image>\n{question}<long_answer>"
            elif i == 0 and section_name == "multiple_choice":
                human_value = f"<image>\n{question}<multiple_choice>"
            elif section_name == "free_response" or section_name == "description":
                human_value = f"{question}<long_answer>"
            elif section_name == "multiple_choice":
                human_value = f"{question}<multiple_choice>"
            else:
                human_value = f"{question}<long_answer>"
            
            current_section_qa_pairs.append({
                "from": "human",
                "value": human_value,
                "type": section_name
            })
            
            # Add gpt answer
            current_section_qa_pairs.append({
                "from": "gpt", 
                "value": answer
            })
        if len(current_section_qa_pairs) > 0:
            targeted_qa_pairs.append(current_section_qa_pairs)
    return targeted_qa_pairs