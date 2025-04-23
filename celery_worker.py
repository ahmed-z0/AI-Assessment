from celery import Celery
import os
from dotenv import load_dotenv
from utils import process_csv_with_ai
import pandas as pd
import json

# Load environment variables
load_dotenv()

# Create Celery app
celery_app = Celery('ai_assessment', 
                    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
                    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'))

# Optional Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour time limit
)

@celery_app.task(bind=True)
def process_csv_task(self, csv_filepath, scoring_sections, name_header_index, model_config):
    """
    Celery task to process CSV with AI scoring
    """
    # Read the CSV to get total number of rows for progress tracking
    df = pd.read_csv(csv_filepath)
    total_rows = len(df)
    
    # Process the CSV with progress updates
    results = []
    
    # Process each row separately and update progress
    for i, (index, row) in enumerate(df.iterrows()):
        # Update task state to track progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': i,
                'total': total_rows,
                'status': f'Processing row {i+1} of {total_rows}'
            }
        )
        
        # Handle individual row processing
        headers = df.columns.tolist()
        candidate_result = process_single_candidate(row, headers, name_header_index, scoring_sections, model_config)
        results.append(candidate_result)
    
    # Return the complete results
    return {
        'status': 'SUCCESS',
        'results': results
    }

def process_single_candidate(row, headers, name_header_index, scoring_sections, model_config):
    """Process a single candidate (row) with AI scoring"""
    candidate_results = {
        'name': row[headers[name_header_index]],
        'sections': []
    }
    
    for section in scoring_sections:
        prompt_template = section['prompt']
        section_name = section.get('section_name', 'Unnamed Section')
        max_marks = section.get('max_marks', 10)
        
        # Replace placeholders in the prompt with actual values from the row
        from utils import replace_placeholders_by_index, get_ai_score
        prompt = replace_placeholders_by_index(prompt_template, row, headers)
        
        # Configure LiteLLM and get score
        from utils import configure_litellm
        configure_litellm(model_config)
        model_string = model_config['model']
        score = get_ai_score(prompt, max_marks, model_string)
        
        # Add section result
        candidate_results['sections'].append({
            'section_name': section_name,
            'score': score,
            'max_marks': max_marks
        })
    
    return candidate_results