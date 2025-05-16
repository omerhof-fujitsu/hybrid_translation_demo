"""
Language detection utility for translation verification.
"""

from typing import Optional
from langdetect import detect, DetectorFactory, LangDetectException

# Set seed for consistent results
DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    """
    Detect the language of a text.
    
    Args:
        text: Text to detect language from
        
    Returns:
        str: ISO 639-1 language code
        
    Raises:
        LangDetectException: If language detection fails
    """
    return detect(text)