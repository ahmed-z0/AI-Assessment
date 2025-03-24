# AI-Assessment

A Flask-based web application for automatically assessing CSV data using AI models. Users can upload a CSV file, define scoring criteria with customized prompts, and receive AI-generated scores in a downloadable CSV file.

## Features

- CSV file upload and header preview
- Dynamic creation of scoring sections
- Customizable prompts with CSV column references
- AI-powered assessment using LiteLLM (supports multiple AI providers)
- Downloadable results

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment (venv)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/AI-Assessment.git
cd AI-Assessment
```

2. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
# source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your API keys:
   - Create a `.env` file based on the provided example
   - Add your API keys for the LLM provider(s) you plan to use

### Running the Application

```bash
python app.py
```

The application will be available at http://127.0.0.1:5000/

## Docker Deployment

You can also run the application using Docker:

```bash
# Build the Docker image
docker build -t ai-assessment .

# Run the container
docker run -p 5000:5000 ai-assessment
```

## Usage Guide

1. **Upload a CSV** - Begin by uploading your CSV file with student/candidate data
2. **Define Scoring Sections** - Create one or more scoring sections by:
   - Setting maximum marks
   - Selecting an output column (or creating a new one)
   - Writing a prompt that references CSV columns using curly braces like `{column_name}`
3. **Process the CSV** - Submit the form to process all rows with AI scoring
4. **Download Results** - The processed CSV with AI-generated scores will be downloaded automatically

## Customization

- Different AI models can be configured in the `utils.py` file
- Modify the prompt structure or scoring approach as needed

## License

[MIT License](LICENSE)
