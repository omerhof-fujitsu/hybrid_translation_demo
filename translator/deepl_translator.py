"""
DeepL API implementation for translation.
"""

import os
from typing import Optional, List, Dict

from .base_translator import BaseTranslator
from utils.logger import logger


class DeepLTranslator(BaseTranslator):
    """
    Translator implementation using DeepL API.
    """
    
    def __init__(self, auth_key: Optional[str] = None, use_math_preservation: bool = True):
        """
        Initialize the DeepL translator.
        
        Args:
            auth_key: DeepL API authentication key
            use_math_preservation: Whether to use math preservation functionality
        """
        super().__init__(use_math_preservation=use_math_preservation)
        
        try:
            # Import the client here to avoid unnecessary dependencies if not used
            import deepl
            
            # Get API key from parameter or environment variable
            self.auth_key = auth_key or os.environ.get("DEEPL_API_KEY")
            
            if not self.auth_key:
                logger.warning("DeepL API key not provided. Please set DEEPL_API_KEY environment variable or pass auth_key parameter.")
                raise ValueError("DeepL API key is required")
            
            # Initialize the DeepL client
            self.translator = deepl.Translator(self.auth_key)
            
            logger.info("DeepL client initialized successfully")
        except ImportError:
            logger.error("deepl is not installed. Install with: pip install deepl")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize DeepL client: {e}")
            raise
    
    def translate(self, text: str, target_language: str) -> str:
        """
        Translate text using DeepL with optional math preservation.
        
        Args:
            text: Text to translate
            target_language: Target language code or name (e.g., 'JA' for Japanese)
            
        Returns:
            str: Translated text
        """
        if not text:
            return text
        
        # Map language names to codes if needed
        language_mapping = {
            'Japanese': 'JA',
            'Hebrew': 'HE',
            'Russian': 'RU',
            'Spanish': 'ES',
            'Chinese': 'ZH',
            'French': 'FR',
            'German': 'DE',
            'Korean': 'KO',
            'Italian': 'IT',
            'Portuguese': 'PT-BR',
            'Arabic': 'AR',
        }
        
        target_code = language_mapping.get(target_language, target_language.upper())
        
        try:
            # If math preservation is enabled, extract mathematical expressions first
            replacements = {}
            modified_text = text
            
            if self.use_math_preservation:
                modified_text, replacements = self.math_preserver.extract_math(text)
            
            # Translate the modified text
            result = self.translator.translate_text(
                modified_text, 
                target_lang=target_code,
                preserve_formatting=True
            )
            translated_text = result.text
            
            # Restore mathematical expressions if math preservation is enabled
            if self.use_math_preservation:
                restored_text = self.math_preserver.restore_math(translated_text, replacements)
                return restored_text
            else:
                return translated_text
            
        except Exception as e:
            logger.error(f"DeepL translation error: {e}")
            # If translation fails, try to restore math expressions in original text
            if self.use_math_preservation and 'replacements' in locals():
                try:
                    return self.math_preserver.restore_math(text, replacements)
                except:
                    pass
            return text  # Return original text as fallback
    
    def batch_translate(self, texts: List[str], target_language: str) -> List[str]:
        """
        Translate a batch of texts using DeepL.
        
        Args:
            texts: List of texts to translate
            target_language: Target language code or name
            
        Returns:
            List[str]: List of translated texts
        """
        if not texts:
            return []
            
        # Process each text individually to preserve math in each
        return [self.translate(text, target_language) for text in texts]