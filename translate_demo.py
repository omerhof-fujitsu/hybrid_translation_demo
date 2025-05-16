#!/usr/bin/env python3
"""
Demo CLI for the Hybrid Translation System.
Supports interactive mode and command-line arguments for text or file translation.
"""

import os
import sys
import json
import click
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm

# Add the parent directory to the path to allow importing from translator and utils
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import translator components
from translator.llm_translator import LLMTranslator
from translator.google_translator import GoogleTranslator
from translator.hybrid_translator import HybridTranslator
try:
    from translator.deepl_translator import DeepLTranslator
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False
    
# Import utilities
from utils.logger import logger
from utils.prompts_manager import PromptsManager

# Load environment variables from .env file
load_dotenv()

# Default dataset type to use for prompts
DEFAULT_DATASET_TYPE = "math"

# Available languages with their codes
LANGUAGES = {
    "japanese": "ja",
    "russian": "ru",
    "hebrew": "he",
    "spanish": "es",
    "chinese": "zh",
    "french": "fr",
    "german": "de",
    "korean": "ko",
    "italian": "it",
    "portuguese": "pt",
    "arabic": "ar",
    "hindi": "hi",
    "bengali": "bn",
    "english": "en"
}

def initialize_translators(dataset_type: str = DEFAULT_DATASET_TYPE):
    """Initialize translator instances based on available API keys."""
    
    # Initialize available translators
    llm_translator = None
    google_translator = None
    deepl_translator = None
    hybrid_translator = None
    
    # Check for LLM API key
    azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if azure_api_key:
        # Use Azure OpenAI
        model_name = os.environ.get("AZURE_OPENAI_MODEL", "azure/attack-gpt4o")
        api_base = os.environ.get("AZURE_OPENAI_API_BASE", "https://llm-sec.openai.azure.com/")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        
        llm_translator = LLMTranslator(
            model_name=model_name,
            api_key=azure_api_key,
            api_base=api_base,
            api_version=api_version,
            dataset_type=dataset_type
        )
    elif openai_api_key:
        # Use regular OpenAI
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")
        
        llm_translator = LLMTranslator(
            model_name=model_name,
            api_key=openai_api_key,
            dataset_type=dataset_type
        )
    else:
        logger.error("No OpenAI API key found. LLM translation will not be available.")
    
    # Check for Google credentials
    google_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if google_creds:
        google_translator = GoogleTranslator(
            api_key_path=google_creds,
            use_math_preservation=(dataset_type == "math")
        )
    else:
        logger.warning("No Google Cloud credentials found. Google translation will not be available.")
    
    # Check for DeepL API key if DeepL is available
    if DEEPL_AVAILABLE:
        deepl_api_key = os.environ.get("DEEPL_API_KEY")
        if deepl_api_key:
            deepl_translator = DeepLTranslator(
                auth_key=deepl_api_key,
                use_math_preservation=(dataset_type == "math")
            )
        else:
            logger.warning("No DeepL API key found. DeepL translation will not be available.")
    
    # Create hybrid translator if possible
    if llm_translator and (google_translator or deepl_translator):
        hybrid_translator = HybridTranslator(
            deepl_translator=deepl_translator,
            google_translator=google_translator,
            llm_translator=llm_translator,
            dataset_type=dataset_type
        )
    elif llm_translator:
        logger.warning("No machine translator available. Only LLM translation will be available.")
    elif google_translator or deepl_translator:
        logger.warning("No LLM translator available. Only machine translation will be available.")
    else:
        logger.error("No translators could be initialized. Please check your API keys.")
        sys.exit(1)
    
    return {
        "llm": llm_translator,
        "google": google_translator,
        "deepl": deepl_translator,
        "hybrid": hybrid_translator
    }

def translate_text(text: str, target_language: str, translator_mode: str, translators: Dict[str, Any]):
    """
    Translate text using the specified translator mode.
    
    Args:
        text: Text to translate
        target_language: Target language name or code
        translator_mode: Which translator to use ('hybrid', 'llm', 'google', 'deepl')
        translators: Dictionary of available translator instances
        
    Returns:
        str: Translated text
    """
    # Normalize language name
    if target_language.lower() in LANGUAGES:
        language_code = LANGUAGES[target_language.lower()]
    else:
        language_code = target_language
    
    # Select translator based on mode
    translator = translators.get(translator_mode.lower())
    
    if not translator:
        available_translators = [k for k, v in translators.items() if v]
        if available_translators:
            translator_mode = available_translators[0]
            translator = translators[translator_mode]
            logger.warning(f"Requested translator '{translator_mode}' not available. Using '{translator_mode}' instead.")
        else:
            logger.error("No translators available. Please check your API keys.")
            sys.exit(1)
    
    # Perform translation
    try:
        logger.info(f"Translating text using {translator_mode} translator to {target_language}...")
        result = translator.translate(text, target_language)
        logger.info("Translation completed successfully.")
        return result
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return None

def display_results(original: str, translated: str, target_language: str):
    """Display translation results in a nice format."""
    
    print("\n" + "="*80)
    print(f"ORIGINAL TEXT (English):")
    print("-"*80)
    print(f"{original}")
    print("\n" + "="*80)
    print(f"TRANSLATED TEXT ({target_language}):")
    print("-"*80)
    print(f"{translated}")
    print("="*80 + "\n")

def load_file(file_path: str) -> str:
    """Load text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        sys.exit(1)

def save_translation(original: str, translated: str, target_language: str, mode: str):
    """Save translation to a file."""
    
    # Create output directory if it doesn't exist
    os.makedirs("translations", exist_ok=True)
    
    # Generate filename
    filename = f"translations/translation_{target_language.lower()}_{mode}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("ORIGINAL TEXT (English):\n")
            f.write("="*80 + "\n")
            f.write(original + "\n\n")
            f.write("TRANSLATED TEXT:\n")
            f.write("="*80 + "\n")
            f.write(translated)
        
        logger.info(f"Translation saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save translation: {e}")

def interactive_mode(translators: Dict[str, Any]):
    """Interactive translation mode."""
    
    available_translators = [k for k, v in translators.items() if v]
    
    print("\n" + "="*80)
    print("HYBRID TRANSLATION SYSTEM - INTERACTIVE MODE")
    print("="*80 + "\n")
    
    print("Available translators:", ", ".join(available_translators))
    print("Available languages:", ", ".join(LANGUAGES.keys()))
    print("\nEnter 'q' or 'quit' to exit.\n")
    
    while True:
        # Get text to translate
        text = input("Enter text to translate (or 'q' to quit): ")
        if text.lower() in ('q', 'quit', 'exit'):
            break
        
        # Get target language
        target_language = input("Enter target language: ")
        if target_language.lower() in ('q', 'quit', 'exit'):
            break
        
        # Get translator mode
        translator_mode = input(f"Enter translator mode ({', '.join(available_translators)}): ") or "hybrid"
        if translator_mode.lower() in ('q', 'quit', 'exit'):
            break
        
        # Validate translator mode
        if translator_mode.lower() not in available_translators:
            print(f"Invalid translator mode. Available options: {', '.join(available_translators)}")
            translator_mode = available_translators[0]
            print(f"Using {translator_mode} instead.")
        
        # Translate text
        translated = translate_text(text, target_language, translator_mode, translators)
        
        if translated:
            display_results(text, translated, target_language)
            
            # Ask if user wants to save the translation
            save_option = input("Save translation to file? (y/n): ")
            if save_option.lower() == 'y':
                save_translation(text, translated, target_language, translator_mode)
        
        print("\n" + "-"*80 + "\n")

@click.command()
@click.option('--text', '-t', help='Text to translate')
@click.option('--file', '-f', help='Input file containing text to translate')
@click.option('--language', '-l', default='Japanese', help='Target language')
@click.option('--mode', '-m', default='hybrid', 
              help='Translation mode (hybrid, llm, google, deepl)')
@click.option('--dataset', '-d', default=DEFAULT_DATASET_TYPE,
              help='Dataset type for prompts (math, gaia, etc.)')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
@click.option('--save', '-s', is_flag=True, help='Save translation to file')
def main(text, file, language, mode, dataset, interactive, save):
    """Hybrid Translation System Demo CLI."""
    
    # Initialize translators
    translators = initialize_translators(dataset)
    
    # Use interactive mode if specified
    if interactive:
        interactive_mode(translators)
        return
    
    # Otherwise, process command-line options
    if file:
        # Load text from file
        text = load_file(file)
    elif not text:
        # Show help if no text is provided
        logger.error("Please provide text to translate using --text or --file, or use --interactive mode.")
        click.echo(main.get_help(click.Context(main)))
        return
    
    # Translate text
    translated = translate_text(text, language, mode, translators)
    
    if translated:
        display_results(text, translated, language)
        
        # Save translation if requested
        if save:
            save_translation(text, translated, language, mode)

if __name__ == '__main__':
    main()