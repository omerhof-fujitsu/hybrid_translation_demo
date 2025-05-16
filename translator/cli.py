#!/usr/bin/env python3
"""
HybridTranslator Demo: Command-line interface for the hybrid translation system.
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import translator components
from translator.hybrid_translator import HybridTranslator
from translator.deepl_translator import DeepLTranslator
from translator.google_translator import GoogleTranslator
from translator.llm_translator import LLMTranslator
from utils.logger import logger

def setup_translators(dataset_type: str, use_google: bool = False):
    """
    Set up and initialize the translators based on available API keys.
    
    Args:
        dataset_type: Type of dataset ('math', 'gaia', 'swe-bench', 'asb')
        use_google: Whether to use Google Translate instead of DeepL
        
    Returns:
        HybridTranslator: Configured translator instance
    """
    # Load API keys from environment
    load_dotenv()
    
    # Configure LLM translator
    azure_key = os.getenv('AZURE_OPENAI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not azure_key and not openai_key:
        print("Error: No LLM API key found. Please set AZURE_OPENAI_API_KEY or OPENAI_API_KEY in your .env file.")
        sys.exit(1)
    
    # If Azure key is available, use Azure model
    if azure_key:
        llm_translator = LLMTranslator(
            model_name="azure/attack-gpt4o",
            api_key=azure_key,
            api_base=os.getenv('AZURE_OPENAI_API_BASE', "https://llm-sec.openai.azure.com/"),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION', "2024-08-01-preview"),
            dataset_type=dataset_type
        )
    else:
        # Fall back to OpenAI
        llm_translator = LLMTranslator(
            model_name="gpt-4o",
            api_key=openai_key,
            dataset_type=dataset_type
        )
    
    # Configure machine translator
    use_math_preservation = (dataset_type == 'math' or dataset_type == 'swe-bench')
    
    if use_google:
        # Google Translate
        google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not google_creds:
            print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set. Google Translate may not work.")
        
        machine_translator = GoogleTranslator(
            api_key_path=google_creds,
            use_math_preservation=use_math_preservation
        )
    else:
        # DeepL
        deepl_key = os.getenv('DEEPL_API_KEY')
        if not deepl_key:
            print("Warning: DEEPL_API_KEY not set. Falling back to Google Translate.")
            # Fall back to Google if DeepL key is not available
            return setup_translators(dataset_type, use_google=True)
        
        machine_translator = DeepLTranslator(
            auth_key=deepl_key,
            use_math_preservation=use_math_preservation
        )
    
    # Create and return hybrid translator
    return HybridTranslator(
        deepl_translator=None if use_google else machine_translator,
        google_translator=machine_translator if use_google else None,
        llm_translator=llm_translator,
        dataset_type=dataset_type
    )

def translate_text(text: str, translator: HybridTranslator, target_language: str) -> str:
    """
    Translate a single text using the hybrid translator.
    
    Args:
        text: Text to translate
        translator: Configured translator instance
        target_language: Target language code or name
        
    Returns:
        str: Translated text
    """
    try:
        return translator.translate(text, target_language)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"[Translation Error: {str(e)}]"

def translate_file(input_file: str, translator: HybridTranslator, 
                  target_language: str, output_file: Optional[str] = None) -> None:
    """
    Translate all text content in a JSON file.
    
    Args:
        input_file: Path to input JSON file
        translator: Configured translator instance
        target_language: Target language code or name
        output_file: Path to output JSON file (if None, prints to stdout)
    """
    try:
        # Load input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine if it's a list or object
        if isinstance(data, list):
            # Process list of objects
            for item in data:
                process_json_item(item, translator, target_language)
        else:
            # Process single object
            process_json_item(data, translator, target_language)
        
        # Write output
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Translated content saved to {output_file}")
        else:
            # Print to stdout
            print(json.dumps(data, ensure_ascii=False, indent=2))
            
    except Exception as e:
        logger.error(f"File translation error: {e}")
        print(f"Error processing file: {e}")

def process_json_item(item: Dict[str, Any], translator: HybridTranslator, target_language: str) -> None:
    """
    Process a JSON object by translating all string values.
    
    Args:
        item: JSON object to process
        translator: Configured translator instance
        target_language: Target language code or name
    """
    for key, value in item.items():
        if isinstance(value, str) and len(value.strip()) > 0:
            # Translate string values
            item[key] = translate_text(value, translator, target_language)
        elif isinstance(value, dict):
            # Recursively process nested objects
            process_json_item(value, translator, target_language)
        elif isinstance(value, list):
            # Process list items
            for i, list_item in enumerate(value):
                if isinstance(list_item, dict):
                    process_json_item(list_item, translator, target_language)
                elif isinstance(list_item, str) and len(list_item.strip()) > 0:
                    value[i] = translate_text(list_item, translator, target_language)

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Hybrid Translator Demo")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', help='Text to translate')
    input_group.add_argument('--file', help='JSON file to translate')
    
    # Translation options
    parser.add_argument('--language', required=True, help='Target language (e.g., Japanese, Hebrew)')
    parser.add_argument('--domain', default='math', choices=['math', 'gaia', 'swe-bench', 'asb'],
                       help='Content domain type')
    parser.add_argument('--google', action='store_true', help='Force using Google Translate')
    
    # Output options
    parser.add_argument('--output', help='Output file for translated content')
    
    args = parser.parse_args()
    
    # Setup translator
    translator = setup_translators(args.domain, use_google=args.google)
    
    if args.text:
        # Translate single text
        result = translate_text(args.text, translator, args.language)
        print("\nTranslation Result:")
        print("-------------------")
        print(result)
    else:
        # Translate file
        translate_file(args.file, translator, args.language, args.output)

if __name__ == "__main__":
    main()