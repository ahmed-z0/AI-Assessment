import os
import pandas as pd
import re
import json
from litellm import completion
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process_csv_with_ai(csv_filepath, scoring_sections, name_header_index, model_config):
    """
    Process a CSV file with AI scoring based on the defined scoring sections using LiteLLM.
    
    Args:
        csv_filepath (str): Path to the input CSV file
        scoring_sections (list): List of dictionaries containing scoring configuration
        name_header_index (int): Index of the column containing candidate names
        model_config (dict): Configuration for the AI model including:
            - model_type: 'default' or 'custom'
            - model: Model identifier string
            - api_key: API key for the service
            - api_base: (Optional) Base URL for custom API endpoints
    
    Returns:
        list: List of dictionaries containing results for each candidate
    """
    # Read the CSV file
    df = pd.read_csv(csv_filepath)
    headers = df.columns.tolist()
    
    # Configure LiteLLM based on the model type
    configure_litellm(model_config)
    
    # Get the model string to use with LiteLLM
    model_string = model_config['model']
    
    # Prepare results data structure
    results = []
    
    # Process each row with the defined scoring sections
    for index, row in df.iterrows():
        candidate_results = {
            'name': row[headers[name_header_index]],
            'sections': []
        }
        
        for section in scoring_sections:
            prompt_template = section['prompt']
            section_name = section.get('section_name', 'Unnamed Section')
            max_marks = section.get('max_marks', 10)  # Default to 10 if not specified
            
            # Replace placeholders in the prompt with actual values from the row
            prompt = replace_placeholders_by_index(prompt_template, row, headers)
            
            # Process with AI model via LiteLLM
            score = get_ai_score(prompt, max_marks, model_string)
            
            # Add section result
            candidate_results['sections'].append({
                'section_name': section_name,
                'score': score,
                'max_marks': max_marks
            })
        
        results.append(candidate_results)
    
    return results

def configure_litellm(model_config):
    """
    Configure LiteLLM based on the model configuration.
    
    Args:
        model_config (dict): Model configuration including type, model name, API key, etc.
    """
    # Reset any previous configuration
    litellm.api_key = None
    litellm.api_base = None
    
    # Set the API key directly on LiteLLM
    litellm.api_key = model_config.get('api_key')
    
    # If it's a custom model with an API base, set that too
    if model_config.get('model_type') == 'custom' and model_config.get('api_base'):
        litellm.api_base = model_config.get('api_base')

def replace_placeholders_by_index(prompt_template, row, headers):
    """
    Replace placeholders in the format {index} with values from the row.
    
    Args:
        prompt_template (str): The prompt template with {index} placeholders
        row (pandas.Series): A row from the DataFrame
        headers (list): List of column headers
    
    Returns:
        str: The prompt with placeholders replaced
    """
    # Find all placeholders in the format {index}
    placeholders = re.findall(r'\{(\d+)\}', prompt_template)
    
    # Replace each placeholder with the corresponding value from the row
    prompt = prompt_template
    for index_str in placeholders:
        try:
            index = int(index_str)
            if 0 <= index < len(headers):
                column_name = headers[index]
                prompt = prompt.replace(f'{{{index}}}', str(row[column_name]))
        except ValueError:
            # If not a valid integer, skip this placeholder
            pass
    
    return prompt

def get_ai_score(prompt, max_marks, model):
    """
    Get a score for the given prompt using LiteLLM.
    
    Args:
        prompt (str): The prompt to send to the AI model
        max_marks (int or float): Maximum marks for this section
        model (str): The model identifier to use with LiteLLM
    
    Returns:
        float: The score assigned by the AI model
    """
    try:
        # First try with structured JSON output
        return get_structured_score(prompt, max_marks, model)
    except Exception as e:
        print(f"Error with structured scoring: {str(e)}")
        # Fall back to text-based scoring if JSON parsing fails
        return get_prompt_score(prompt, max_marks, model)

def get_structured_score(prompt, max_marks, model):
    """
    Get a score using JSON structured output.
    
    Args:
        prompt (str): The prompt to send to the AI model
        max_marks (int or float): Maximum marks for this section
        model (str): The model identifier to use with LiteLLM
    
    Returns:
        float: The score assigned by the AI model
    """
    # Construct the system and user messages
    system_message = f"""You are an assessment AI. Your task is to evaluate answers based on prompts and provide a score between 0 and {max_marks}.
    Your evaluation should be fair, consistent, and based on the content of the answer.
    You must return a JSON object with a 'score' property containing only the numeric score. For example: {{"score": 8.5}}"""
    
    user_message = f"""Based on the following input, provide a score between 0 and {max_marks}:
    
    {prompt}"""
    
    try:
        # Call the model with JSON object format
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=100
        )
        
        # Extract the JSON result
        result = json.loads(response.choices[0].message.content)
        score = result.get('score', 0)
        
        # Ensure the score is within the valid range
        score = max(0, min(float(max_marks), score))
        return round(score, 2)  # Round to 2 decimal places
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"Error parsing JSON response: {str(e)}")
        # If there's an issue with the JSON response, try to extract score from text
        return extract_score_from_text(response.choices[0].message.content, max_marks)
    except Exception as e:
        # For all other errors, fall back to prompt-based scoring
        print(f"Error with structured output: {str(e)}")
        return get_prompt_score(prompt, max_marks, model)

def get_prompt_score(prompt, max_marks, model):
    """
    Get a score using traditional prompt engineering.
    
    Args:
        prompt (str): The prompt to send to the AI model
        max_marks (int or float): Maximum marks for this section
        model (str): The model identifier to use with LiteLLM
    
    Returns:
        float: The score assigned by the AI model
    """
    # Construct the full prompt with instructions
    full_prompt = f"""
    Based on the following input, provide a score between 0 and {max_marks}.
    Return only a numeric value (or a float with up to 2 decimal places).
    
    Input: {prompt}
    
    Score (0-{max_marks}):
    """
    
    try:
        # Call the AI model
        response = completion(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3,  # Lower temperature for more consistent scoring
            max_tokens=10     # We only need a short response
        )
        
        # Extract the score from the response
        score_text = response.choices[0].message.content.strip()
        return extract_score_from_text(score_text, max_marks)
    except Exception as e:
        print(f"Error with prompt-based scoring: {str(e)}")
        return 0  # Return 0 in case of errors

def extract_score_from_text(text, max_marks):
    """
    Extract a numeric score from text response.
    
    Args:
        text (str): The text to extract a score from
        max_marks (int or float): Maximum marks for this section
    
    Returns:
        float: The extracted score
    """
    # Try to convert to float directly
    try:
        score = float(text)
        # Ensure the score is within the valid range
        score = max(0, min(float(max_marks), score))
        return round(score, 2)  # Round to 2 decimal places
    except ValueError:
        # If direct conversion fails, look for digits in the response
        digits = re.findall(r'\d+\.?\d*', text)
        if digits:
            try:
                score = float(digits[0])
                score = max(0, min(float(max_marks), score))
                return round(score, 2)
            except ValueError:
                pass
        
        # If all else fails, return 0
        return 0