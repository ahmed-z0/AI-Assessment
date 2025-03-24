import os
import pandas as pd
import re
import json
from litellm import completion, get_supported_openai_params
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

def process_csv_with_ai(csv_filepath, scoring_sections, name_header_index, model, api_key, supports_json_output):
    """
    Process a CSV file with AI scoring based on the defined scoring sections.
    
    Args:
        csv_filepath (str): Path to the input CSV file
        scoring_sections (list): List of dictionaries containing scoring configuration
            Each dict should have:
            - prompt: The prompt template with {index} placeholders
            - output_column: The column name where scores will be written
            - max_marks: Maximum marks for this section
            - section_name: Name of the scoring section
        name_header_index (int): Index of the column containing candidate names
        model (str): The AI model to use
        api_key (str): API key for the model provider
        supports_json_output (bool): Whether the model supports structured output
    
    Returns:
        list: List of dictionaries containing results for each candidate
    """
    # Read the CSV file
    df = pd.read_csv(csv_filepath)
    headers = df.columns.tolist()
    
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
            
            # Process with AI model
            score = get_ai_score(prompt, max_marks, model, api_key, supports_json_output)
            
            # Add section result
            candidate_results['sections'].append({
                'section_name': section_name,
                'score': score,
                'max_marks': max_marks
            })
        
        results.append(candidate_results)
    
    return results

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

def get_ai_score(prompt, max_marks, model, api_key, supports_json_output):
    """
    Get a score for the given prompt using an AI model.
    
    Args:
        prompt (str): The prompt to send to the AI model
        max_marks (int or float): Maximum marks for this section
        model (str): The AI model to use
        api_key (str): API key for the model provider
        supports_json_output (bool): Whether the model supports structured output
    
    Returns:
        float: The score assigned by the AI model
    """
    try:
        # Set the appropriate provider based on model prefix
        provider = None
        
        # Determine the provider based on the model name
        if model.startswith('gpt-'):
            os.environ["OPENAI_API_KEY"] = api_key
            provider = "openai"
        elif model.startswith('claude-'):
            os.environ["ANTHROPIC_API_KEY"] = api_key
            provider = "anthropic"
        elif model.startswith('gemini-'):
            os.environ["GOOGLE_API_KEY"] = api_key
            provider = "google"
        else:
            # Set as custom model/provider
            os.environ["OPENAI_API_KEY"] = api_key  # Default to OpenAI
        
        if supports_json_output:
            # Use structured output format for supported models
            return get_structured_score(prompt, max_marks, model, provider)
        else:
            # Use traditional prompting for models without structured output support
            return get_prompt_score(prompt, max_marks, model, provider)
    
    except Exception as e:
        print(f"Error calling AI model: {str(e)}")
        return 0  # Return 0 in case of errors

def get_structured_score(prompt, max_marks, model, provider=None):
    """
    Get a score using structured output JSON format.
    
    Args:
        prompt (str): The prompt to send to the AI model
        max_marks (int or float): Maximum marks for this section
        model (str): The AI model to use
        provider (str, optional): The provider to use (openai, anthropic, etc.)
    
    Returns:
        float: The score assigned by the AI model
    """
    # Define the JSON schema for the response
    json_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "description": f"A score between 0 and {max_marks} based on the assessment criteria"
            }
        },
        "required": ["score"]
    }
    
    # Construct the system and user messages
    system_message = f"""You are an assessment AI. Your task is to evaluate answers based on prompts and provide a score between 0 and {max_marks}.
    Your evaluation should be fair, consistent, and based on the content of the answer.
    You must return a JSON object with a 'score' property containing only the numeric score."""
    
    user_message = f"""Based on the following input, provide a score between 0 and {max_marks}:
    
    {prompt}"""
    
    # Call the AI model using litellm with structured output format
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_schema", "schema": json_schema},
        temperature=0.3,
        max_tokens=100,
        custom_llm_provider=provider
    )
    
    # Extract the JSON result
    try:
        result = json.loads(response.choices[0].message.content)
        score = result.get('score', 0)
        
        # Ensure the score is within the valid range
        score = max(0, min(float(max_marks), score))
        return round(score, 2)  # Round to 2 decimal places
    except (json.JSONDecodeError, AttributeError, KeyError):
        # If there's an issue with the JSON response, fall back to prompt-based scoring
        return get_prompt_score(prompt, max_marks, model, provider)

def get_prompt_score(prompt, max_marks, model, provider=None):
    """
    Get a score using traditional prompt engineering.
    
    Args:
        prompt (str): The prompt to send to the AI model
        max_marks (int or float): Maximum marks for this section
        model (str): The AI model to use
        provider (str, optional): The provider to use (openai, anthropic, etc.)
    
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
    
    # Call the AI model using litellm
    response = completion(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.3,  # Lower temperature for more consistent scoring
        max_tokens=10,    # We only need a short response
        custom_llm_provider=provider
    )
    
    # Extract the score from the response
    score_text = response.choices[0].message.content.strip()
    
    # Try to convert to float
    try:
        score = float(score_text)
        # Ensure the score is within the valid range
        score = max(0, min(float(max_marks), score))
        return round(score, 2)  # Round to 2 decimal places
    except ValueError:
        # If conversion fails, look for digits in the response
        digits = re.findall(r'\d+\.?\d*', score_text)
        if digits:
            try:
                score = float(digits[0])
                score = max(0, min(float(max_marks), score))
                return round(score, 2)
            except ValueError:
                pass
        
        # If all else fails, return 0
        return 0