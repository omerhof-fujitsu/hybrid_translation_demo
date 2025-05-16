"""
Translator package for the hybrid translation system.
"""

# Import translator classes for easy access
from .base_translator import BaseTranslator
from .llm_translator import LLMTranslator
from .google_translator import GoogleTranslator

# Try to import DeepL translator if available
try:
    from .deepl_translator import DeepLTranslator
    __all__ = ['BaseTranslator', 'LLMTranslator', 'GoogleTranslator', 'DeepLTranslator', 'HybridTranslator']
except ImportError:
    __all__ = ['BaseTranslator', 'LLMTranslator', 'GoogleTranslator', 'HybridTranslator']

# Always import hybrid translator last as it depends on the others
from .hybrid_translator import HybridTranslator