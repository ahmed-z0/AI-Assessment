from celery import Celery
import os
from dotenv import load_dotenv
from utils import process_csv_with_ai, configure_litellm, replace_placeholders_by_index, get_ai_score
import pandas as pd
import json
import traceback

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
    worker_max_tasks_per_child=10, # Restart worker after 10 tasks
    worker_prefetch_multiplier=1, # Prefetch only one task at a time
    task_acks_late=True, # Acknowledge task after it's completed
    task_reject_on_worker_lost=True, # Reject tasks if the worker is terminated
)

@celery_app.task(bind=True)
def process_csv_task(self, csv_filepath, scoring_sections, name_header_index, model_config):
    """
    Celery task to process CSV with AI scoring
    """
    # Update task state to STARTED
    self.update_state(
        state='STARTED',
        meta={
            'current': 0,
            'total': 1,
            'status': 'Reading CSV file...'
        }
    )
    
    try:
        # Read the CSV to get total number of rows for progress tracking
        df = pd.read_csv(csv_filepath)
        total_rows = len(df)
        headers = df.columns.tolist()
        
        # Update task state with total rows
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 0,
                'total': total_rows,
                'status': f'Starting processing {total_rows} rows...'
            }
        )
        
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
                    'status': f'Processing row {i+1} of {total_rows}: {row[headers[name_header_index]]}'
                }
            )
            
            # Handle individual row processing
            try:
                candidate_result = process_single_candidate(row, headers, name_header_index, scoring_sections, model_config)
                results.append(candidate_result)
            except Exception as e:
                # Log the error but continue processing
                error_msg = f"Error processing candidate {row[headers[name_header_index]]}: {str(e)}"
                print(error_msg)
                
                # Add a placeholder with error message
                candidate_results = {
                    'name': row[headers[name_header_index]],
                    'sections': [{'section_name': section['section_name'], 
                                'score': 0, 
                                'max_marks': section['max_marks']} 
                                for section in scoring_sections],
                    'error': str(e)
                }
                results.append(candidate_results)
        
        # Return the complete results
        return {
            'status': 'SUCCESS',
            'results': results
        }
    
    except Exception as e:
        # Capture full stack trace
        stack_trace = traceback.format_exc()
        error_message = f"Failed to process CSV: {str(e)}\n{stack_trace}"
        print(error_message)
        
        # Re-raise the exception with more info
        # This will mark the task as failed
        raise Exception(error_message)

def process_single_candidate(row, headers, name_header_index, scoring_sections, model_config):
    """Process a single candidate (row) with AI scoring"""
    candidate_results = {
        'name': row[headers[name_header_index]],
        'sections': []
    }
    
    # Configure LiteLLM for this processing
    configure_litellm(model_config)
    model_string = model_config['model']
    
    for section in scoring_sections:
        prompt_template = section['prompt']
        section_name = section.get('section_name', 'Unnamed Section')
        max_marks = section.get('max_marks', 10)
        
        # Replace placeholders in the prompt with actual values from the row
        prompt = replace_placeholders_by_index(prompt_template, row, headers)
        
        # Process with AI model to get score
        score = get_ai_score(prompt, max_marks, model_string)
        
        # Add section result
        candidate_results['sections'].append({
            'section_name': section_name,
            'score': score,
            'max_marks': max_marks
        })
    
    return candidate_results