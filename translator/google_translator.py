"""
Google Cloud Translation API implementation.
"""

import os
from typing import Optional, List, Dict

from .base_translator import BaseTranslator
from utils.logger import logger


class GoogleTranslator(BaseTranslator):
    """
    Translator implementation using Google Cloud Translation API.
    """
    
    def __init__(self, api_key_path: Optional[str] = None, use_math_preservation: bool = True):
        """
        Initialize the Google translator.
        
        Args:
            api_key_path: Path to Google Cloud API key JSON file
            use_math_preservation: Whether to use math preservation functionality
        """
        super().__init__(use_math_preservation=use_math_preservation)
        
        try:
            # Set credentials environment variable if provided
            if api_key_path:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = api_key_path
                
            # Import the client here to avoid unnecessary dependencies if not used
            from google.cloud import translate_v2 as translate
            self.client = translate.Client()
            
            logger.info("Google Translate client initialized successfully")
        except ImportError:
            logger.error("google-cloud-translate is not installed. "
                         "Install with: pip install google-cloud-translate")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Google Translate client: {e}")
            raise
    
    def translate(self, text: str, target_language: str) -> str:
        """
        Translate text using Google Translate with optional math preservation.
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'ja' for Japanese)
            
        Returns:
            str: Translated text
        """
        if not text:
            return text
        
        # Map language names to codes if needed
        language_mapping = {
            'Japanese': 'ja',
            'Hebrew': 'he',  # Google uses 'he' for Hebrew
            'Russian': 'ru',
            'Spanish': 'es',
            'Chinese': 'zh',
            'French': 'fr',
            'German': 'de',
            'Korean': 'ko',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Arabic': 'ar',
            'Hindi': 'hi',
            'Bengali': 'bn',
            'English': 'en'
        }
        
        target_code = language_mapping.get(target_language, target_language)
        
        try:
            # If math preservation is enabled, extract mathematical expressions first
            replacements = {}
            modified_text = text
            
            if self.use_math_preservation:
                modified_text, replacements = self.math_preserver.extract_math(text)
            
            # Translate the modified text
            result = self.client.translate(modified_text, target_language=target_code)
            translated_text = result['translatedText']
            
            # Restore mathematical expressions if math preservation is enabled
            if self.use_math_preservation:
                restored_text = self.math_preserver.restore_math(translated_text, replacements)
                return restored_text
            else:
                return translated_text
            
        except Exception as e:
            logger.error(f"Google translation error: {e}")
            # If translation fails, try to restore math expressions in original text
            if self.use_math_preservation and 'replacements' in locals():
                try:
                    return self.math_preserver.restore_math(text, replacements)
                except:
                    pass
            return text  # Return original text as fallback
    
    def batch_translate(self, texts: List[str], target_language: str) -> List[str]:
        """
        Translate a batch of texts using Google Translate.
        Uses the built-in batch functionality for efficiency.
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            
        Returns:
            List[str]: List of translated texts
        """
        if not texts:
            return []
            
        # Process each text individually to preserve math in each
        return [self.translate(text, target_language) for text in texts]