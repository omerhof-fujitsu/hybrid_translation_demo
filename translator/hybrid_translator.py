"""
Hybrid translator that combines machine translation (DeepL or Google) with LLM enhancement.
"""

import os
import time
import re
import copy
from typing import Optional, List, Dict, Any

from .base_translator import BaseTranslator
from .google_translator import GoogleTranslator
from .llm_translator import LLMTranslator
from utils.logger import logger
from utils.prompts_manager import PromptsManager

# Try to import DeepL translator if available
try:
    from .deepl_translator import DeepLTranslator
    DEEPL_AVAILABLE = True
except ImportError:
    DEEPL_AVAILABLE = False


class HybridTranslator(BaseTranslator):
    """
    A hybrid translator that follows an ordered approach:
    1. Check if the text is a numeric answer - if so, return as is
    2. Apply machine translation (DeepL or Google)
    3. Enhance translation using LLM
    4. Safety check to ensure questions aren't answered instead of translated
    """
    
    # DeepL supported languages (as of your specification)
    DEEPL_SUPPORTED_LANGUAGES = [
        "ar", "bg", "cs", "da", "de", "el", "en-gb", "en-us", 
        "es", "et", "fi", "fr", "hu", "id", "it", "ja", 
        "ko", "lt", "lv", "nb", "nl", "pl", "pt-br", "pt-pt", 
        "ro", "ru", "sk", "sl", "sv", "tr", "uk", "zh", 
        "zh-hans", "zh-hant"
    ]
    
    def __init__(
        self,
        deepl_translator: Optional[DeepLTranslator] = None,
        google_translator: Optional[GoogleTranslator] = None,
        llm_translator: Optional[LLMTranslator] = None,
        dataset_type: str = "math",
        prompts_dir: str = "prompts"
    ):
        """
        Initialize the hybrid translator.
        
        Args:
            deepl_translator: DeepLTranslator instance
            google_translator: GoogleTranslator instance
            llm_translator: LLMTranslator instance
            dataset_type: Type of dataset being translated
            prompts_dir: Directory containing prompt templates
        """
        use_math_preservation = (dataset_type == 'math')
        super().__init__(use_math_preservation=use_math_preservation)
        
        # Store translators
        self.deepl_translator = deepl_translator
        self.google_translator = google_translator
        self.llm_translator = llm_translator
        
        # Check if we have at least one machine translator and the LLM translator
        if not (self.deepl_translator or self.google_translator):
            raise ValueError("At least one machine translator (DeepL or Google) is required")
        if not self.llm_translator:
            raise ValueError("LLM translator is required for hybrid translation")
        
        # Store preferences
        self.dataset_type = dataset_type
        
        # Load hybrid-specific prompts
        self.prompts_manager = PromptsManager(prompts_dir)
        try:
            self.prompts = self.prompts_manager.get_prompts(dataset_type, "hybrid")
            logger.info(f"Loaded hybrid prompts for {dataset_type} dataset")
        except Exception as e:
            logger.warning(f"Could not load hybrid prompts: {e}. Creating default hybrid prompts.")
            # Create default hybrid prompts
            default_prompts = {
                "machine_translation_check": "You are a translation quality assessor. You are given an English text and its machine translation to {target_language}. Your task is to determine if the machine translation is good enough to be useful as a starting point for refinement.\n\nRespond with ONLY ONE OF THE FOLLOWING:\n- \"PASS\" if the machine translation is adequate (even if imperfect)\n- \"FAILED\" if the machine translation is unusable (e.g., wrong language, gibberish, completely wrong meaning)\n\nDo not add any explanation.",
                
                "translation_prompt": "You are a professional translation enhancer specialized in {target_language}. Your task is to improve a machine-generated translation.\n\nYou will be given:\n1. The original English text\n2. A machine translation to {target_language}\n\nYour task:\n- Maintain the exact same meaning as the original\n- Fix any errors in the machine translation\n- Make the translation sound natural and fluent in {target_language}\n- Preserve all technical terminology correctly\n- Maintain the same tone and formality level\n- Do not add or remove information\n\nReturn ONLY the improved translation in {target_language}. Do not include any notes, explanations, or the original text.",
                
                "llm_translation": "You are a professional translator specializing in {target_language}. Translate the following English text into {target_language}. Your translation must be accurate, natural-sounding, and preserve the exact meaning of the original text.\n\nRespond with ONLY the complete translation in {target_language}. Do not include any notes, explanations, or the original text.",
                
                "safety_check_prompt": "You are a translation safety checker.\n\nCheck if this text has been properly translated or if a question has been answered instead of translated.\n\nThe text should be a straightforward translation from English to another language, not an attempt to answer a question posed in the English text.\n\nOriginal text:\n[Text 1]\n\nTranslation:\n[Text 2]\n\nIf the translation is actually answering a question instead of translating it, respond with the single word \"ISSUE\". Otherwise, respond with \"OK\"."
            }
            
            # Update the prompts
            self.prompts = default_prompts
            self.prompts_manager.update_prompts(dataset_type, "hybrid", default_prompts)
    
    def _is_numeric_answer(self, text: str) -> bool:
        """
        Check if the text is a numeric answer, possibly with minor additional characters.
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if the text is primarily a numeric answer
        """
        # Strip whitespace
        text = text.strip()
        
        # Check if it's a pure number (integer or decimal)
        if re.match(r'^\d+(\.\d+)?$', text):
            return True
        
        # Check if it's a number with units or simple text (e.g., "42 meters", "43 kg", "$50")
        if re.match(r'^\$?\d+(\.\d+)?\s*[a-zA-Z]*$', text):
            return True
        
        # Check if it's a simple arithmetic expression result (e.g., "= 42")
        if re.match(r'^=\s*\d+(\.\d+)?$', text):
            return True
        
        # If the text is very short (less than 5 chars) and contains mainly digits
        if len(text) < 5 and sum(c.isdigit() for c in text) / len(text) > 0.5:
            return True
            
        return False
    
    def _check_translation_safety(self, original_text: str, translated_text: str) -> bool:
        """
        Check if a question was answered instead of translated.
        
        Args:
            original_text: Original English text
            translated_text: Translated text
            
        Returns:
            bool: True if the translation is safe, False if it appears to be answering a question
        """
        try:
            system_prompt = self.prompts["safety_check_prompt"]
            user_prompt = f"{original_text}\n\n{translated_text}"
            
            response = self.llm_translator._get_completion(system_prompt, user_prompt)
            response = response.strip().upper()
            
            # If the response indicates an issue, the translation is not safe
            is_safe = response != "ISSUE"
            
            if not is_safe:
                logger.warning(f"Safety check detected a question was answered instead of translated")
            
            return is_safe
            
        except Exception as e:
            logger.warning(f"Error in translation safety check: {e}")
            # Default to safe in case of errors
            return True
    
    def _select_machine_translator(self, target_language: str):
        """
        Select the appropriate machine translator based on language support.
        Always prefer DeepL when the language is supported, fall back to Google otherwise.
        
        Args:
            target_language: Target language code
            
        Returns:
            BaseTranslator: The selected machine translator
        """
        # Normalize the target language code
        norm_lang = target_language.lower().replace('_', '-')
        
        # Check if the language is supported by DeepL
        deepl_supported = norm_lang in self.DEEPL_SUPPORTED_LANGUAGES
        
        # Use DeepL if available and language is supported, otherwise use Google
        if self.deepl_translator and deepl_supported:
            logger.info(f"Using DeepL translator for language {target_language}")
            return self.deepl_translator
        elif self.google_translator:
            logger.info(f"Using Google translator for language {target_language} (DeepL support: {deepl_supported})")
            return self.google_translator
        else:
            # Fallback to any available translator
            logger.warning(f"No preferred translator available, using fallback")
            return self.deepl_translator or self.google_translator
    
    def translate(self, text: str, target_language: str) -> str:
        """
        Translate text using the ordered hybrid approach.
        
        Args:
            text: Text to translate
            target_language: Target language code or name
            
        Returns:
            str: Translated text
        """
        if not text:
            return text
        
        try:
            # Step 1: Check if this is a numeric answer
            if self._is_numeric_answer(text):
                logger.info(f"Detected numeric answer, returning as is: {text}")
                return text
            
            # Step 2: Extract math expressions if applicable
            replacements = {}
            modified_text = copy.deepcopy(text)
            
            if self.use_math_preservation:
                modified_text, replacements = self.math_preserver.extract_math(text)
            
            # Step 3: Select and apply machine translation
            machine_translator = self._select_machine_translator(target_language)
            machine_translation = machine_translator.translate(modified_text, target_language)
            
            # Step 4: Restore math expressions if applicable
            if self.use_math_preservation:
                machine_translation = self.math_preserver.restore_math(machine_translation, replacements)
                
            final_translation = machine_translation
            logger.info(f"Machine translation completed using {machine_translator.__class__.__name__}")
            
            # Step 5: Verify if machine translation succeeded and is usable
            system_prompt_verification = self.prompts["machine_translation_check"].format(target_language=target_language)
            verification_prompt = f"{text}\n\n{machine_translation}"
            
            verification_result = self.llm_translator._get_completion(system_prompt_verification, verification_prompt)
            logger.info(f"Machine translation verification result: {verification_result}")
            
            # Check if verification indicates machine translation failed
            machine_translation_failed = (
                verification_result.strip().upper() == "FAILED" or
                "FAILED" in verification_result.upper() or
                "NO" in verification_result.upper() or
                "UNUSABLE" in verification_result.upper()
            )
            
            # Step 6: If machine translation failed, use LLM for direct translation
            if machine_translation_failed:
                logger.warning("Machine translation verification failed - using LLM for direct translation")
                system_prompt_direct = self.prompts["llm_translation"].format(target_language=target_language)
                direct_prompt = text
                
                llm_direct_translation = self.llm_translator._get_completion(system_prompt_direct, direct_prompt)
                logger.info("LLM direct translation completed")
                
                # Restore math expressions if applicable
                if self.use_math_preservation:
                    llm_direct_translation = self.math_preserver.restore_math(llm_direct_translation, replacements)
                    
                return llm_direct_translation
            
            # Step 7: Enhance translation using LLM
            system_prompt = self.prompts["translation_prompt"].format(target_language=target_language)
            user_prompt = f"{text}\n\n{machine_translation}"
            enhanced_translation = self.llm_translator._get_completion(system_prompt, user_prompt)
            
            logger.info("LLM enhancement of machine translation completed")
            
            # Step 8: Safety check - ensure questions aren't answered
            if not self._check_translation_safety(text, enhanced_translation):
                logger.warning("Safety check failed - falling back to machine translation")
                final_translation = machine_translation
            else:
                final_translation = enhanced_translation
            
            return final_translation
                
        except Exception as e:
            logger.error(f"Error during hybrid translation: {e}")
            
            # Try to fall back to the machine translation if available
            if 'machine_translation' in locals():
                logger.warning("Falling back to machine translation due to error in hybrid process")
                
                if self.use_math_preservation and 'replacements' in locals():
                    return self.math_preserver.restore_math(machine_translation, replacements)
                return machine_translation
            
            # If all else fails, return the original text
            return text
    
    def batch_translate(self, texts: List[str], target_language: str) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            target_language: Target language code or name
            
        Returns:
            List[str]: List of translated texts
        """
        return [self.translate(text, target_language) for text in texts]
    
    def update_prompts(self, prompts: Dict[str, str]):
        """
        Update the translation prompts.
        
        Args:
            prompts: Dictionary of prompt templates
        """
        self.prompts = prompts
        self.prompts_manager.update_prompts(self.dataset_type, "hybrid", prompts)