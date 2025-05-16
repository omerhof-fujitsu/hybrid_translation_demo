#!/usr/bin/env python3
"""
Example usage of the Hybrid Translation System.
This script showcases different translation examples.
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the translator package
from translator.llm_translator import LLMTranslator
from translator.google_translator import GoogleTranslator
from translator.hybrid_translator import HybridTranslator
from utils.logger import logger

# Load environment variables
load_dotenv()

# Sample texts to translate
SAMPLES = {
    "simple": "Hello, how are you today?",
    
    "math": "Solve the quadratic equation $ax^2 + bx + c = 0$ using the quadratic formula $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.",
    
    "technical": "The neural network architecture consists of three hidden layers with ReLU activation functions, followed by a softmax output layer for multi-class classification.",
    
    "complex": """Consider a right triangle with sides a, b, and hypotenuse c. 
According to the Pythagorean theorem, $a^2 + b^2 = c^2$. 
If we know that $a = 3$ and $b = 4$, what is the value of $c$?
Express your answer as a decimal approximation to two places."""
}

def run_examples():
    """Run translation examples using different translators and languages."""
    
    # Check for API keys
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    google_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not openai_key:
        logger.error("No OpenAI API key found. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY in .env")
        sys.exit(1)
    
    if not google_creds:
        logger.warning("No Google credentials found. Only LLM translator will be available.")
    
    # Initialize translators
    llm_translator = LLMTranslator(
        model_name=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        api_key=openai_key,
        dataset_type="math"
    )
    
    google_translator = None
    if google_creds:
        google_translator = GoogleTranslator(
            api_key_path=google_creds,
            use_math_preservation=True
        )
    
    hybrid_translator = None
    if google_translator:
        hybrid_translator = HybridTranslator(
            google_translator=google_translator,
            llm_translator=llm_translator,
            dataset_type="math"
        )
    
    # Define target languages
    languages = ["Japanese", "Spanish", "Russian"]
    
    # Run examples
    for language in languages:
        print(f"\n{'='*80}\nTranslating to {language}\n{'='*80}")
        
        for sample_name, sample_text in SAMPLES.items():
            print(f"\nSample: {sample_name}")
            print(f"Original: {sample_text}")
            
            # LLM translation
            llm_result = llm_translator.translate(sample_text, language)
            print(f"\nLLM Translation:\n{llm_result}")
            
            # Google translation (if available)
            if google_translator:
                google_result = google_translator.translate(sample_text, language)
                print(f"\nGoogle Translation:\n{google_result}")
            
            # Hybrid translation (if available)
            if hybrid_translator:
                hybrid_result = hybrid_translator.translate(sample_text, language)
                print(f"\nHybrid Translation:\n{hybrid_result}")
            
            print(f"\n{'-'*80}")

if __name__ == "__main__":
    run_examples()