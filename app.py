import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import json
from utils import process_csv_with_ai

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

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
            
            # Get model and API key from the form
            model = request.form.get('model')
            api_key = request.form.get('api_key')
            
            # If model is 'custom', get the custom model name
            if model == 'custom':
                model = request.form.get('custom_model')
                if not model:
                    flash('Please enter a custom model name')
                    return redirect(request.url)
            
            if not api_key:
                flash('Please enter an API key')
                return redirect(request.url)
            
            # Get the name header from the form
            name_header_index = request.form.get('name_header_index')
            if not name_header_index:
                flash('Please select a name header')
                return redirect(request.url)
            
            # Get CSV headers to display in the form
            df = pd.read_csv(filepath)
            headers = df.columns.tolist()
            
            # Get the scoring sections from the form
            scoring_sections = json.loads(request.form.get('scoring_sections', '[]'))
            
            if not scoring_sections:
                flash('No scoring sections defined')
                return redirect(request.url)
            
            # Process the CSV with AI scoring
            try:
                # Define which models support structured output
                structured_output_models = [
                    "gpt-3.5-turbo", "gpt-4o", 
                    "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229",
                    "gemini-1.5-pro", "gemini-1.5-flash-preview"
                ]
                
                # Check if the selected model supports structured output
                supports_json_output = model in structured_output_models
                
                # Process the CSV with AI scoring
                results_data = process_csv_with_ai(
                    filepath, 
                    scoring_sections, 
                    int(name_header_index),
                    model,
                    api_key,
                    supports_json_output
                )
                
                # Store results in session for the results page
                session['results'] = results_data
                session['headers'] = headers
                
                # Redirect to results page
                return redirect(url_for('results'))
            except Exception as e:
                flash(f'Error processing CSV: {str(e)}')
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload a CSV file.')
            return redirect(request.url)
    
    # GET request - render the upload form
    return render_template('index.html')

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

@app.route('/results')
def results():
    # Get results from session
    results_data = session.get('results')
    headers = session.get('headers')
    
    if not results_data:
        flash('No results available. Please process a CSV file first.')
        return redirect(url_for('index'))
    
    return render_template('results.html', results=results_data, headers=headers)

if __name__ == '__main__':
    app.run(debug=True)