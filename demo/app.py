"""
Demo web application for the HybridTrans translation system.
"""

import os
import json
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
import logging

# Load environment variables from .env file
load_dotenv()

# Import translator components
from translator import (
    HybridTranslator, 
    DeepLTranslator, 
    GoogleTranslator, 
    LLMTranslator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
AVAILABLE_LANGUAGES = [
    {"code": "ja", "name": "Japanese"},
    {"code": "he", "name": "Hebrew"},
    {"code": "ru", "name": "Russian"},
    {"code": "es", "name": "Spanish"},
    {"code": "fr", "name": "French"},
    {"code": "de", "name": "German"},
    {"code": "zh", "name": "Chinese"},
    {"code": "ko", "name": "Korean"},
    {"code": "ar", "name": "Arabic"},
    {"code": "it", "name": "Italian"},
    {"code": "pt", "name": "Portuguese"},
    {"code": "hi", "name": "Hindi"},
    {"code": "ro", "name": "Romanian"}
]

DATASET_TYPES = [
    {"value": "math", "name": "Mathematical Content"},
    {"value": "gaia", "name": "General Content"},
    {"value": "swe-bench", "name": "Code & Technical"},
    {"value": "asb", "name": "Academic Content"}
]

def create_translator():
    """
    Create and configure the hybrid translator based on environment variables.
    """
    # Get API keys from environment
    DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY", "")
    GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://llm-sec.openai.azure.com/")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "azure/attack-gpt4o")
    
    # Initialize translators based on available credentials
    llm_translator = None
    deepl_translator = None
    google_translator = None
    
    try:
        # Initialize LLM translator (required)
        llm_translator = LLMTranslator(
            model_name=AZURE_OPENAI_MODEL,
            api_key=AZURE_OPENAI_API_KEY,
            api_base=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            dataset_type="math"  # Default dataset type
        )
        logger.info("LLM Translator initialized successfully")
        
        # Initialize DeepL translator if API key is available
        if DEEPL_AUTH_KEY:
            deepl_translator = DeepLTranslator(auth_key=DEEPL_AUTH_KEY)
            logger.info("DeepL Translator initialized successfully")
        
        # Initialize Google translator if credentials are available
        if GOOGLE_CREDENTIALS_PATH:
            google_translator = GoogleTranslator(api_key_path=GOOGLE_CREDENTIALS_PATH)
            logger.info("Google Translator initialized successfully")
        
        # Use at least one machine translator
        if not deepl_translator and not google_translator:
            logger.warning("No machine translator credentials provided. Using fallback.")
            # For demo purposes, we'll initialize GoogleTranslator without API key
            # It will fail gracefully and return the original text
            google_translator = GoogleTranslator()
        
        # Create and return hybrid translator
        return HybridTranslator(
            deepl_translator=deepl_translator,
            google_translator=google_translator,
            llm_translator=llm_translator,
            dataset_type="math"  # Default dataset type
        )
    
    except Exception as e:
        logger.error(f"Error initializing translators: {e}")
        return None

# Create translator instance
translator = create_translator()

@app.route('/')
def index():
    """Render the main page."""
    return render_template(
        'index.html',
        languages=AVAILABLE_LANGUAGES,
        dataset_types=DATASET_TYPES
    )

@app.route('/translate', methods=['POST'])
def translate():
    """API endpoint for translation."""
    try:
        # Get request data
        data = request.get_json()
        text = data.get('text', '')
        target_language = data.get('language', 'Japanese')
        dataset_type = data.get('dataset_type', 'math')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Update translator dataset type if needed
        if translator.dataset_type != dataset_type:
            translator.dataset_type = dataset_type
            # Also update LLM translator
            translator.llm_translator.dataset_type = dataset_type
        
        # Translate text
        translated_text = translator.translate(text, target_language)
        
        return jsonify({
            'translated_text': translated_text,
            'dataset_type': dataset_type,
            'target_language': target_language
        })
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/sample_prompts')
def sample_prompts():
    """Return sample prompts for the demo."""
    samples = {
        "math": "Solve for x: 2x + 3 = 7 where x ∈ ℝ. Then find the value of x².",
        "gaia": "Climate change is affecting ecosystems worldwide. Scientists observe that rising temperatures are causing shifts in animal migration patterns.",
        "swe-bench": "```python\ndef calculate_average(numbers):\n    total = sum(numbers)\n    return total / len(numbers)\n```\nThis function calculates the average of a list of numbers.",
        "asb": "The study examined the effects of sleep deprivation on cognitive performance. Participants who slept less than 6 hours performed significantly worse on memory tasks."
    }
    return jsonify(samples)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)