import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
import json
from utils import process_csv_with_ai
from flask_session import Session
from celery_worker import celery_app, process_csv_task

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

# Configure server-side session storage
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"  # Alternatives: "redis", "memcached", etc.

# Initialize the extension
Session(app)

# Configure file upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select a file, browser submits an empty file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get model configuration from the form
            model_config = get_model_config_from_form(request.form)
            
            if not model_config.get('api_key'):
                flash('Please enter an API key')
                return redirect(request.url)
            
            # Get the name header from the form
            name_header_index = request.form.get('name_header_index')
            if not name_header_index:
                flash('Please select a name header')
                return redirect(request.url)
            
            # Get the scoring sections from the form
            scoring_sections = json.loads(request.form.get('scoring_sections', '[]'))
            
            if not scoring_sections:
                flash('No scoring sections defined')
                return redirect(request.url)
            
            # Submit the job to the Celery task queue instead of processing synchronously
            try:
                # Read CSV headers for later retrieval
                df = pd.read_csv(filepath)
                headers = df.columns.tolist()
                
                # Store headers in a redis cache or db (simplified approach: using session for headers only)
                session['headers'] = headers
                
                # Submit the task to Celery
                task = process_csv_task.delay(
                    filepath, 
                    scoring_sections, 
                    int(name_header_index),
                    model_config
                )
                
                # Redirect to the progress page with task_id as URL parameter
                return redirect(url_for('task_progress', task_id=task.id))
                
            except Exception as e:
                flash(f'Error starting processing: {str(e)}')
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload a CSV file.')
            return redirect(request.url)
    
    # GET request - render the upload form
    return render_template('index.html')

@app.route('/progress/<task_id>')
def task_progress(task_id):
    """Show a page that monitors the task progress via AJAX"""
    if not task_id:
        flash('No task ID provided. Please start a new assessment.')
        return redirect(url_for('index'))
    
    # Verify task exists
    task = process_csv_task.AsyncResult(task_id)
    if not task:
        flash('Invalid task ID. Please start a new assessment.')
        return redirect(url_for('index'))
    
    return render_template('progress.html', task_id=task_id)

@app.route('/task_status/<task_id>')
def task_status(task_id):
    """API endpoint to check the status of a task"""
    task = process_csv_task.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        # Job has not started yet
        response = {
            'state': task.state,
            'status': 'Pending...',
            'current': 0,
            'total': 100,
            'percent': 0
        }
    elif task.state == 'PROGRESS':
        # Job is in progress
        response = {
            'state': task.state,
            'status': task.info.get('status', ''),
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 100),
            'percent': int(task.info.get('current', 0) / task.info.get('total', 100) * 100)
        }
    elif task.state == 'SUCCESS':
        # Job completed successfully
        response = {
            'state': task.state,
            'status': 'Complete!',
            'current': 100,
            'total': 100,
            'percent': 100,
            'result_url': url_for('results', task_id=task_id)
        }
    else:
        # Something unexpected happened
        response = {
            'state': task.state,
            'status': str(task.info),
            'current': 0,
            'total': 100,
            'percent': 0
        }
    
    return jsonify(response)

def get_model_config_from_form(form_data):
    """
    Extract model configuration from form data.
    
    Args:
        form_data: The form data from the request
        
    Returns:
        dict: The model configuration
    """
    model_type = form_data.get('model_type', 'default')
    model_config = {
        'model_type': model_type
    }
    
    if model_type == 'default':
        model_config['model'] = form_data.get('default_model', 'openai/gpt-4o')
        model_config['api_key'] = form_data.get('api_key', '')
    else:  # custom
        model_config['model'] = form_data.get('custom_model_name', '')
        model_config['api_key'] = form_data.get('custom_api_key', '')
        model_config['api_base'] = form_data.get('custom_api_base', '')
    
    return model_config

@app.route('/preview_headers', methods=['POST'])
def preview_headers():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return json.dumps({'error': 'No file part'})
    
    file = request.files['file']
    
    # If user does not select a file, browser submits an empty file
    if file.filename == '':
        return json.dumps({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get CSV headers
        try:
            df = pd.read_csv(filepath)
            headers = df.columns.tolist()
            return json.dumps({'headers': headers, 'filename': filename})
        except Exception as e:
            return json.dumps({'error': f'Error reading CSV: {str(e)}'})
    else:
        return json.dumps({'error': 'File type not allowed. Please upload a CSV file.'})

@app.route('/results/<task_id>')
def results(task_id):
    # Get task result directly from Celery
    task = process_csv_task.AsyncResult(task_id)
    
    if not task or task.state != 'SUCCESS':
        flash('Results not available. Please wait for processing to complete.')
        return redirect(url_for('task_progress', task_id=task_id))
    
    # Get results from the task
    results_data = task.result.get('results', [])
    # Get headers from session (this is one piece of data we're still using from session)
    headers = session.get('headers', [])
    
    if not results_data:
        flash('No results available. Processing may have failed.')
        return redirect(url_for('index'))
    
    return render_template('results.html', results=results_data, headers=headers, task_id=task_id)

if __name__ == '__main__':
    app.run(debug=True)