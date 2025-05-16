"""
Base translator class for all translation implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple, Any
from utils.math_preserver import SimpleMathPreserver
from utils.logger import logger

class BaseTranslator(ABC):
    """
    Abstract base class for all translators.
    Defines the common interface that all translator implementations must follow.
    """
    
    def __init__(self, use_math_preservation: bool = True):
        """
        Initialize the base translator.
        
        Args:
            use_math_preservation: Whether to use math preservation functionality
        """
        self.use_math_preservation = use_math_preservation
        if use_math_preservation:
            self.math_preserver = SimpleMathPreserver()
    
    @abstractmethod
    def translate(self, text: str, target_language: str) -> str:
        """
        Translate text from English to the target language.
        
        Args:
            text: Text to translate
            target_language: Target language code or name
            
        Returns:
            str: Translated text
        """
        pass
    
    def batch_translate(self, texts: List[str], target_language: str) -> List[str]:
        """
        Translate a batch of texts.
        Default implementation calls translate() for each text.
        Subclasses can override this for more efficient batch processing.
        
        Args:
            texts: List of texts to translate
            target_language: Target language code or name
            
        Returns:
            List[str]: List of translated texts
        """
        return [self.translate(text, target_language) for text in texts]