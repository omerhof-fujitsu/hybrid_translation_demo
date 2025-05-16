"""
Batch processor for translating large datasets efficiently.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import concurrent.futures

from .hybrid_translator import HybridTranslator
from .deepl_translator import DeepLTranslator
from .google_translator import GoogleTranslator
from .llm_translator import LLMTranslator
from utils.logger import logger

class BatchProcessor:
    """
    Batch processor for translating large datasets efficiently.
    Includes parallel processing, error handling, and progress tracking.
    """
    
    def __init__(
        self,
        dataset_type: str = "math",
        target_language: str = "Japanese",
        use_google: bool = False,
        max_workers: int = 4,
        azure_model: str = "azure/attack-gpt4o",
        openai_model: str = "gpt-4o"
    ):
        """
        Initialize the batch processor.
        
        Args:
            dataset_type: Type of dataset ('math', 'gaia', 'swe-bench', 'asb')
            target_language: Target language code or name
            use_google: Whether to use Google Translate instead of DeepL
            max_workers: Maximum number of parallel workers
            azure_model: Azure OpenAI model name to use if available
            openai_model: OpenAI model name to use as fallback
        """
        self.dataset_type = dataset_type
        self.target_language = target_language
        self.use_google = use_google
        self.max_workers = max_workers
        self.azure_model = azure_model
        self.openai_model = openai_model
        
        # Create translator
        self.translator = self._setup_translator()
        
        # Statistics
        self.stats = {
            "total_items": 0,
            "successful": 0,
            "failed": 0,
            "start_time": None,
            "end_time": None
        }
    
    def _setup_translator(self) -> HybridTranslator:
        """
        Set up and initialize the translators based on available API keys.
        
        Returns:
            HybridTranslator: Configured translator instance
        """
        # Configure LLM translator
        azure_key = os.getenv('AZURE_OPENAI_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        if not azure_key and not openai_key:
            raise ValueError("No LLM API key found. Please set AZURE_OPENAI_API_KEY or OPENAI_API_KEY.")
        
        # If Azure key is available, use Azure model
        if azure_key:
            llm_translator = LLMTranslator(
                model_name=self.azure_model,
                api_key=azure_key,
                api_base=os.getenv('AZURE_OPENAI_API_BASE', "https://llm-sec.openai.azure.com/"),
                api_version=os.getenv('AZURE_OPENAI_API_VERSION', "2024-08-01-preview"),
                dataset_type=self.dataset_type
            )
        else:
            # Fall back to OpenAI
            llm_translator = LLMTranslator(
                model_name=self.openai_model,
                api_key=openai_key,
                dataset_type=self.dataset_type
            )
        
        # Configure machine translator
        use_math_preservation = (self.dataset_type == 'math' or self.dataset_type == 'swe-bench')
        
        if self.use_google:
            # Google Translate
            google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            machine_translator = GoogleTranslator(
                api_key_path=google_creds,
                use_math_preservation=use_math_preservation
            )
        else:
            # DeepL
            deepl_key = os.getenv('DEEPL_API_KEY')
            if not deepl_key:
                # Fall back to Google if DeepL key is not available
                self.use_google = True
                return self._setup_translator()
            
            machine_translator = DeepLTranslator(
                auth_key=deepl_key,
                use_math_preservation=use_math_preservation
            )
        
        # Create and return hybrid translator
        return HybridTranslator(
            deepl_translator=None if self.use_google else machine_translator,
            google_translator=machine_translator if self.use_google else None,
            llm_translator=llm_translator,
            dataset_type=self.dataset_type
        )
    
    def _translate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a single item including all its string fields.
        
        Args:
            item: Item to translate
            
        Returns:
            Dict[str, Any]: Translated item
        """
        translated_item = {}
        
        for key, value in item.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                # Translate string values
                try:
                    translated_item[key] = self.translator.translate(value, self.target_language)
                except Exception as e:
                    logger.error(f"Error translating field '{key}': {e}")
                    translated_item[key] = value  # Use original value on error
            elif isinstance(value, dict):
                # Recursively translate nested dictionaries
                translated_item[key] = self._translate_item(value)
            elif isinstance(value, list):
                # Translate list items
                translated_item[key] = self._translate_list(value)
            else:
                # Keep non-string values as is
                translated_item[key] = value
        
        return translated_item
    
    def _translate_list(self, items: List[Any]) -> List[Any]:
        """
        Translate a list of items.
        
        Args:
            items: List of items to translate
            
        Returns:
            List[Any]: Translated list
        """
        translated_list = []
        
        for item in items:
            if isinstance(item, str) and len(item.strip()) > 0:
                # Translate string values
                try:
                    translated_list.append(self.translator.translate(item, self.target_language))
                except Exception as e:
                    logger.error(f"Error translating list item: {e}")
                    translated_list.append(item)  # Use original value on error
            elif isinstance(item, dict):
                # Recursively translate dictionaries
                translated_list.append(self._translate_item(item))
            elif isinstance(item, list):
                # Recursively translate nested lists
                translated_list.append(self._translate_list(item))
            else:
                # Keep non-string values as is
                translated_list.append(item)
        
        return translated_list
    
    def process_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of items for translation.
        
        Args:
            data: List of items to translate
            
        Returns:
            List[Dict[str, Any]]: Translated items
        """
        self.stats["total_items"] = len(data)
        self.stats["successful"] = 0
        self.stats["failed"] = 0
        self.stats["start_time"] = time.time()
        
        translated_data = []
        
        # Use a progress bar to show translation progress
        with tqdm(total=len(data), desc="Translating items") as pbar:
            # Limit the number of parallel workers based on LLM API rate limits
            if self.max_workers > 1:
                # Use parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_idx = {executor.submit(self._translate_item, item): i 
                                     for i, item in enumerate(data)}
                    
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            translated_item = future.result()
                            translated_data.append(translated_item)
                            self.stats["successful"] += 1
                        except Exception as e:
                            logger.error(f"Error processing item {idx}: {e}")
                            # Use original item on error
                            translated_data.append(data[idx])
                            self.stats["failed"] += 1
                        
                        pbar.update(1)
            else:
                # Use sequential processing
                for item in data:
                    try:
                        translated_item = self._translate_item(item)
                        translated_data.append(translated_item)
                        self.stats["successful"] += 1
                    except Exception as e:
                        logger.error(f"Error processing item: {e}")
                        # Use original item on error
                        translated_data.append(item)
                        self.stats["failed"] += 1
                    
                    pbar.update(1)
        
        self.stats["end_time"] = time.time()
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        # Print statistics
        print(f"\nBatch processing completed:")
        print(f"  Total items: {self.stats['total_items']}")
        print(f"  Successfully translated: {self.stats['successful']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Average time per item: {duration / self.stats['total_items']:.2f} seconds")
        
        return translated_data
    
    def process_file(self, input_file: str, output_file: str) -> None:
        """
        Process a file containing items for translation.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file
        """
        try:
            # Load input file
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process data
            if isinstance(data, list):
                translated_data = self.process_batch(data)
            else:
                # Handle single item or dictionary with nested lists
                translated_data = self._translate_item(data)
            
            # Write output file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            
            print(f"Translated data saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise