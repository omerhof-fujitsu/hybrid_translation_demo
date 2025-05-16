"""
LLM-based translator using LiteLLM with a 3-step QA and correction pipeline.
Includes language detection verification.
"""

import os
import time
from typing import Optional, List, Dict, Any, Tuple

from .base_translator import BaseTranslator
from utils.logger import logger
from utils.prompts_manager import PromptsManager

try:
    # Attempt to import language detection
    from langdetect import detect, LangDetectException
    LANG_DETECT_AVAILABLE = True
except ImportError:
    LANG_DETECT_AVAILABLE = False
    logger.warning("langdetect not installed. Language verification will be disabled.")
    # Define placeholder for LangDetectException
    class LangDetectException(Exception):
        pass


class LLMTranslator(BaseTranslator):
    """
    Translator implementation using LiteLLM for various LLM providers,
    with a 3-step translation QA and correction pipeline.
    Includes language detection verification.
    """
    
    def __init__(
        self,
        model_name: str = "azure/attack-gpt4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = "https://llm-sec.openai.azure.com/",
        api_version: str = "2024-08-01-preview",
        dataset_type: str = "math",
        prompts_dir: str = "prompts"
    ):
        """
        Initialize the LLM translator.
        
        Args:
            model_name: The model to use for translation
            api_key: API key for the LLM service
            api_base: Base URL for the API
            api_version: API version
            dataset_type: Type of dataset ('math' or 'gaia')
            prompts_dir: Directory containing prompt templates
        """
        super().__init__(use_math_preservation=(dataset_type == 'math'))
        
        # Store dataset type for prompting
        self.dataset_type = dataset_type
        
        try:
            # Import litellm here to avoid unnecessary dependencies if not used
            from litellm import completion
            
            # Store the completion function
            self.completion = completion
            
            # Get API key from parameter or environment variable
            self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            
            if not self.api_key:
                logger.warning("API key not provided. Please set AZURE_OPENAI_API_KEY or OPENAI_API_KEY environment variable or pass api_key parameter.")
                # Use a default value - this will be replaced with actual key in production
                self.api_key = "x"
            
            # Store model configuration
            self.model_name = model_name
            self.api_base = api_base
            self.api_version = api_version
            
            # Initialize prompts manager and load prompts
            self.prompts_manager = PromptsManager(prompts_dir)
            self.prompts = self.prompts_manager.get_prompts(dataset_type, "llm")
            
            # Language code mapping for verification
            self.language_code_map = {
                'ja': 'ja',  # Japanese
                'japanese': 'ja',
                'ru': 'ru',  # Russian
                'russian': 'ru',
                'he': 'he',  # Hebrew
                'hebrew': 'he',
                'es': 'es',  # Spanish
                'spanish': 'es',
                'ro': 'ro',  # Romanian
                'romanian': 'ro',
                'ko': 'ko',  # Korean
                'korean': 'ko',
                'zh': 'zh-cn',  # Chinese
                'chinese': 'zh-cn',
                'fr': 'fr',  # French
                'french': 'fr',
                'de': 'de',  # German
                'german': 'de',
                'it': 'it',  # Italian
                'italian': 'it',
                'pt': 'pt',  # Portuguese
                'portuguese': 'pt',
                'ar': 'ar',  # Arabic
                'arabic': 'ar',
                'hi': 'hi',  # Hindi
                'hindi': 'hi',
                'bn': 'bn',  # Bengali
                'bengali': 'bn',
                'en': 'en',  # English
                'english': 'en'
            }
            
            logger.info(f"LLM Translator initialized successfully with model: {model_name} for dataset type: {dataset_type}")
            
        except ImportError as e:
            logger.error("litellm package not installed. Install with: pip install litellm")
            raise ImportError("litellm package not installed. Install with: pip install litellm")
        except Exception as e:
            logger.error(f"Failed to initialize LLM translator: {e}")
            raise
    
    def _get_completion(self, system_prompt: str, user_prompt: str) -> str:
        """
        Get completion from the LLM using LiteLLM.
        
        Args:
            system_prompt: The system prompt to instruct the model
            user_prompt: The prompt to send to the LLM
            
        Returns:
            str: The LLM's response
        """
        try:
            # Set up the API parameters - account for different model providers
            api_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            }
            
            # Add provider-specific parameters
            if "azure" in self.model_name.lower():
                api_params.update({
                    "api_key": self.api_key,
                    "api_base": self.api_base,
                    "api_version": self.api_version
                })
            else:
                api_params.update({
                    "api_key": self.api_key
                })
            
            # Call the LLM
            response = self.completion(**api_params)
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error during LLM completion: {e}")
            return f"Error: {str(e)}"
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of text.
        
        Args:
            text: Text to detect language
            
        Returns:
            str: Detected language code
            
        Raises:
            LangDetectException: If language detection fails
        """
        if LANG_DETECT_AVAILABLE:
            return detect(text)
        else:
            # Return an empty string if langdetect is not available
            logger.warning("Language detection attempted but langdetect is not installed")
            raise LangDetectException("langdetect not installed")
    
    def _verify_language(self, text: str, target_language: str) -> bool:
        """
        Verify that the text is in the target language.
        
        Args:
            text: Text to verify
            target_language: Target language code or name
            
        Returns:
            bool: True if the text is in the target language, False otherwise
        """
        if not LANG_DETECT_AVAILABLE:
            # Skip verification if langdetect is not available
            logger.warning("Language verification skipped: langdetect not installed")
            return True
            
        if not text or len(text.strip()) < 10:
            # Too short to reliably detect language
            logger.warning("Text too short for reliable language detection")
            return True
        
        # Get the language code for detection
        target_code = self.language_code_map.get(target_language.lower(), target_language.lower())
        
        try:
            # Detect the language of the text
            detected_lang = self.detect_language(text)
            
            # Check if the detected language matches the target language
            if detected_lang == target_code:
                logger.info(f"Language verified: detected {detected_lang}, expected {target_code}")
                return True
            else:
                logger.warning(f"Language mismatch: detected {detected_lang}, expected {target_code}")
                return False
                
        except LangDetectException as e:
            logger.warning(f"Language detection error: {e}")
            # Return True to not block the process on language detection failures
            return True
    
    def _three_step_translation(self, text: str, target_language: str) -> str:
        """
        Perform a 3-step translation QA and correction pipeline.
        
        Args:
            text: Text to translate
            target_language: Target language name
            
        Returns:
            str: Final translated text
        """
        max_retries = 3
        lang_emphasis_added = False
        
        for attempt in range(max_retries):
            try:
                # Step 1: Initial Translation
                system_prompt_1 = self.prompts["system_prompt_step1"].format(target_language=target_language)
                
                # Add emphasis if language detection failed previously
                if lang_emphasis_added:
                    emphasis = f"IMPORTANT: You MUST respond ONLY in {target_language}. Do not use any other language in your response."
                    system_prompt_1 = f"{emphasis}\n\n{system_prompt_1}"
                
                initial_translation = self._get_completion(system_prompt_1, text)
                
                if self.dataset_type != 'math' and LANG_DETECT_AVAILABLE:
                    # Verify the language of the translation
                    correct_language = self._verify_language(initial_translation, target_language)
                
                    if not correct_language:
                        # If language verification failed, retry with emphasis
                        if not lang_emphasis_added:
                            lang_emphasis_added = True
                            logger.info(f"Language verification failed. Retrying with emphasis on {target_language}")
                            continue
                        else:
                            # If already tried with emphasis, log warning and continue anyway
                            logger.warning(f"Language verification failed even with emphasis. Continuing with the process.")
                
                # Step 2: Review Translation
                system_prompt_2 = self.prompts["system_prompt_step2"]
                review_prompt = f"Original English Text:\n{text}\n\nTranslated Text:\n{initial_translation}"
                review_feedback = self._get_completion(system_prompt_2, review_prompt)
                
                # Check if the review found any issues
                if not review_feedback.strip():
                    # If no issues were found, return the initial translation directly
                    logger.info("Review found no issues with the translation. Skipping correction step.")
                    return initial_translation
                
                # If there are issues, attempt to correct
                system_prompt_3 = self.prompts["system_prompt_step3"]
                
                # Add language emphasis if needed
                if lang_emphasis_added:
                    emphasis = f"IMPORTANT: You MUST respond ONLY in {target_language}. Do not use any other language in your response."
                    system_prompt_3 = f"{emphasis}\n\n{system_prompt_3}"
                
                correction_prompt = f"Original English Text:\n{text}\n\nPrevious Translation:\n{initial_translation}\n\nReviewer Feedback:\n{review_feedback}"
                final_translation = self._get_completion(system_prompt_3, correction_prompt)
                
                # Verify the language of the final translation
                if self.dataset_type != 'math' and LANG_DETECT_AVAILABLE:
                    if not self._verify_language(final_translation, target_language):
                        # If language verification failed, use the initial translation as fallback
                        logger.warning(f"Language verification failed for the final translation. Using initial translation as fallback.")
                        return initial_translation
                
                return final_translation
                
            except Exception as e:
                logger.warning(f"Translation QA pipeline error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 + attempt)  # Exponential backoff
                else:
                    logger.error(f"Translation QA pipeline failed after {max_retries} attempts: {e}")
                    return text  # Return original text if all attempts fail
        
        return text  # Fallback to original text
    
    def translate(self, text: str, target_language: str) -> str:
        """
        Translate text using the LLM with a 3-step QA pipeline.
        
        Args:
            text: Text to translate
            target_language: Target language name (e.g., 'Japanese', 'Hindi')
            
        Returns:
            str: Translated text
        """
        if not text:
            return text
        
        try:
            # If math preservation is enabled, extract and protect math expressions
            if self.use_math_preservation:
                modified_text, replacements = self.math_preserver.extract_math(text)
                translated_text = self._three_step_translation(modified_text, target_language)
                # Restore math expressions in the translated text
                return self.math_preserver.restore_math(translated_text, replacements)
            else:
                # Translate without math preservation
                return self._three_step_translation(text, target_language)
                        
        except Exception as e:
            logger.error(f"Error during translation process: {e}")
            return text  # Return original text if any error occurs
    
    def batch_translate(self, texts: List[str], target_language: str) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            target_language: Target language
            
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
        self.prompts_manager.update_prompts(self.dataset_type, "llm", prompts)