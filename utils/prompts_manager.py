"""
Utility to manage system prompts for different datasets and translators.
"""

import os
import json
from typing import Dict, Any
# from .utils import logger
from utils.logger import logger

# Default directory where prompt templates are stored
DEFAULT_PROMPTS_DIR = "prompts"

class PromptsManager:
    """
    Manages loading and retrieving system prompts for different datasets and translators.
    """
    
    def __init__(self, prompts_dir: str = DEFAULT_PROMPTS_DIR):
        """
        Initialize the PromptsManager.
        
        Args:
            prompts_dir: Directory containing prompt template JSON files
        """
        self.prompts_dir = prompts_dir
        self.prompts = {}
        
        # Create prompts directory if it doesn't exist
        os.makedirs(prompts_dir, exist_ok=True)
        
        # Try to load existing prompts
        self._load_all_prompts()
        
        # Initialize with default prompts if not loaded
        if not self.prompts:
            self._initialize_default_prompts()
    
    def _load_all_prompts(self):
        """Load all prompt template JSON files from the prompts directory."""
        try:
            if not os.path.exists(self.prompts_dir):
                logger.warning(f"Prompts directory does not exist: {self.prompts_dir}")
                return
                
            dataset_types = [d for d in os.listdir(self.prompts_dir) 
                            if os.path.isdir(os.path.join(self.prompts_dir, d))]
            
            for dataset_type in dataset_types:
                dataset_dir = os.path.join(self.prompts_dir, dataset_type)
                prompt_files = [f for f in os.listdir(dataset_dir) 
                                if f.endswith('.json')]
                
                for prompt_file in prompt_files:
                    translator_type = os.path.splitext(prompt_file)[0]
                    file_path = os.path.join(dataset_dir, prompt_file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            prompts = json.load(f)
                            
                        if dataset_type not in self.prompts:
                            self.prompts[dataset_type] = {}
                            
                        self.prompts[dataset_type][translator_type] = prompts
                        logger.info(f"Loaded prompts for {dataset_type}/{translator_type}")
                    except Exception as e:
                        logger.error(f"Failed to load prompts from {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
    
    def _initialize_default_prompts(self):
        """Initialize with default prompts."""
        # Math dataset prompts
        math_llm_prompts = {
            "system_prompt_step1": """You are a professional translator specialized in academic content and mathematics. Your task is to translate math problems from English to {target_language} for high school students.
                For each problem, follow these guidelines:
                Preserve all LaTeX expressions exactly as written (do not translate or alter math symbols, equations, or formatting).
                Maintain the original meaning faithfully — the translation must accurately reflect the logical and mathematical structure.
                Use fluent, natural {target_language} that is clear and appropriate for a high school audience. Avoid overly technical or formal phrasing unless necessary for clarity.
                If the English sentence includes instructions or questions, ensure the tone is educational and polite.
                Do not attempt to solve or simplify the math problem — only translate the text.
                Return only the translated problem in {target_language}. No additional commentary or formatting changes.""",
            
            "system_prompt_step2": """You are a bilingual academic reviewer specialized in math education.
                You are given an English math question and its translated version. Compare the two and check whether the translated version has any major issues.
                Only report problems that are:
                - **Meaning-altering inaccuracies**
                - **Missing instructional parts** (e.g., question type, constraints, diagram references)
                - **Missing or altered math-related expressions or formatting**
                If you find any such issues, describe them briefly in plain text.  
                If the translation is complete and faithful, say **nothing at all**.
                Do not re-translate. Do not explain your role. Only return issue descriptions if applicable.""",
            
            "system_prompt_step3": """You are a professional translator and reviewer of academic math content.
                You will revise a previously translated math question based on reviewer feedback, which describes inaccuracies or missing elements compared to the original English.
                Your task is to:
                - **Apply only the necessary corrections** based on the feedback
                - **Preserve all LaTeX expressions** and formatting exactly as in the original translation
                - **Avoid paraphrasing or rewriting anything else**
                Only output the corrected translation. Do not explain, comment, or add any formatting."""
        }
        
        # General text dataset prompts
        general_llm_prompts = {
            "system_prompt_step1": """You are a professional translator. Your task is to translate all input text from English to {target_language}.
                Always respond only with the translated version in fluent {target_language}. Even if the input is a question or contains specific terminology.
                Do not answer or solve questions — just translate them exactly.
                Return only the translated text in {target_language}. No additional commentary or formatting changes.""",
            
            "system_prompt_step2": """You are a bilingual reviewer.
                You are given an English text and its translated version. Compare the two and check whether the translated version has any major issues.
                Only report problems that are:
                - **Meaning-altering inaccuracies**
                - **Missing parts**
                If you find any such issues, describe them briefly in plain text.  
                If the translation is complete and faithful, say **nothing at all**.
                Do not re-translate. Do not explain your role. Only return issue descriptions if applicable.""",
            
            "system_prompt_step3": """You are a professional translator and reviewer.
                You will revise a previously translated text based on reviewer feedback, which describes inaccuracies or missing elements compared to the original English.
                Your task is to:
                - **Apply only the necessary corrections** based on the feedback
                - **Avoid paraphrasing or rewriting anything else**
                Only output the corrected translation. Do not explain, comment, or add any formatting."""
        }

        # Technical text dataset prompts
        technical_llm_prompts = {
            "system_prompt_step1": """You are a professional translator specialized in technical and scientific content. Your task is to translate technical text from English to {target_language}.
                For technical documents, follow these guidelines:
                Preserve all technical terminology, using the standard terms in {target_language} when they exist.
                Maintain all code, variable names, and technical symbols exactly as written.
                Translate acronyms only if they have standard translations in {target_language}. Otherwise, keep the English acronym and provide the full translation in parentheses the first time it appears.
                Use clear, precise {target_language} that sounds natural to technical readers in that language.
                Maintain the same level of formality and technical precision as the original text.
                Return only the translated text in {target_language}. No additional commentary or formatting changes.""",
            
            "system_prompt_step2": """You are a bilingual technical reviewer.
                You are given an English technical document and its translated version. Compare the two and check whether the translated version has any major issues.
                Only report problems that are:
                - **Meaning-altering inaccuracies**
                - **Mistranslated technical terms**
                - **Missing parts or explanations**
                - **Inconsistent translation of technical terminology**
                If you find any such issues, describe them briefly in plain text.  
                If the translation is complete and faithful, say **nothing at all**.
                Do not re-translate. Do not explain your role. Only return issue descriptions if applicable.""",
            
            "system_prompt_step3": """You are a professional technical translator and reviewer.
                You will revise a previously translated technical document based on reviewer feedback, which describes inaccuracies or missing elements compared to the original English.
                Your task is to:
                - **Apply only the necessary corrections** based on the feedback
                - **Ensure technical terminology is consistent throughout**
                - **Preserve all code, symbols, and variables exactly**
                - **Avoid paraphrasing or rewriting unaffected parts**
                Only output the corrected translation. Do not explain, comment, or add any formatting."""
        }
        
        # Create default hybrid prompts
        math_hybrid_prompts = {
            "machine_translation_check": """You are a translation quality assessor. You are given an English text containing mathematical content and its machine translation to {target_language}. Your task is to determine if the machine translation is good enough to be useful as a starting point for refinement.

Respond with ONLY ONE OF THE FOLLOWING:
- "PASS" if the machine translation is adequate (even if imperfect)
- "FAILED" if the machine translation is unusable (e.g., wrong language, gibberish, completely wrong meaning)

Do not add any explanation.""",
            
            "translation_prompt": """You are a professional translation enhancer specialized in mathematical content in {target_language}. Your task is to improve a machine-generated translation.

You will be given:
1. The original English text with mathematical content
2. A machine translation to {target_language}

Your task:
- Maintain the exact same meaning as the original
- Fix any errors in the machine translation
- Make the translation sound natural and fluent in {target_language}
- Preserve all LaTeX expressions and mathematical notation exactly as is
- Maintain the same tone and formality level
- Do not add or remove information
- Do not attempt to solve or simplify the math problems

Return ONLY the improved translation in {target_language}. Do not include any notes, explanations, or the original text.""",
            
            "llm_translation": """You are a professional translator specializing in mathematical content in {target_language}. Translate the following English text into {target_language}. Your translation must be accurate, natural-sounding, and preserve the exact meaning of the original text.

Important guidelines:
- Preserve all LaTeX expressions and mathematical notation exactly as written
- Use appropriate mathematical terminology in {target_language}
- Maintain the same level of formality and clarity

Respond with ONLY the complete translation in {target_language}. Do not include any notes, explanations, or the original text.""",
            
            "safety_check_prompt": """You are a translation safety checker for mathematical content.

Check if this math problem has been properly translated or if a question has been answered instead of translated.

The text should be a straightforward translation from English to another language, not an attempt to solve or answer a mathematical question posed in the English text.

Original text:
[Text 1]

Translation:
[Text 2]

If the translation is actually answering a question instead of translating it, respond with the single word "ISSUE". Otherwise, respond with "OK"."""
        }
        
        general_hybrid_prompts = {
            "machine_translation_check": """You are a translation quality assessor. You are given an English text and its machine translation to {target_language}. Your task is to determine if the machine translation is good enough to be useful as a starting point for refinement.

Respond with ONLY ONE OF THE FOLLOWING:
- "PASS" if the machine translation is adequate (even if imperfect)
- "FAILED" if the machine translation is unusable (e.g., wrong language, gibberish, completely wrong meaning)

Do not add any explanation.""",
            
            "translation_prompt": """You are a professional translation enhancer specialized in {target_language}. Your task is to improve a machine-generated translation.

You will be given:
1. The original English text
2. A machine translation to {target_language}

Your task:
- Maintain the exact same meaning as the original
- Fix any errors in the machine translation
- Make the translation sound natural and fluent in {target_language}
- Preserve all technical terminology correctly
- Maintain the same tone and formality level
- Do not add or remove information

Return ONLY the improved translation in {target_language}. Do not include any notes, explanations, or the original text.""",
            
            "llm_translation": """You are a professional translator specializing in {target_language}. Translate the following English text into {target_language}. Your translation must be accurate, natural-sounding, and preserve the exact meaning of the original text.

Respond with ONLY the complete translation in {target_language}. Do not include any notes, explanations, or the original text.""",
            
            "safety_check_prompt": """You are a translation safety checker.

Check if this text has been properly translated or if a question has been answered instead of translated.

The text should be a straightforward translation from English to another language, not an attempt to answer a question posed in the English text.

Original text:
[Text 1]

Translation:
[Text 2]

If the translation is actually answering a question instead of translating it, respond with the single word "ISSUE". Otherwise, respond with "OK"."""
        }
        
        technical_hybrid_prompts = {
            "machine_translation_check": """You are a translation quality assessor for technical content. You are given an English technical text and its machine translation to {target_language}. Your task is to determine if the machine translation is good enough to be useful as a starting point for refinement.

Respond with ONLY ONE OF THE FOLLOWING:
- "PASS" if the machine translation is adequate (even if imperfect)
- "FAILED" if the machine translation is unusable (e.g., wrong language, gibberish, completely wrong meaning, mistranslated technical terms)

Do not add any explanation.""",
            
            "translation_prompt": """You are a professional translation enhancer specialized in technical content in {target_language}. Your task is to improve a machine-generated translation.

You will be given:
1. The original English technical text
2. A machine translation to {target_language}

Your task:
- Maintain the exact same meaning as the original
- Fix any errors in the machine translation
- Ensure all technical terminology is translated correctly and consistently
- Make the translation sound natural and fluent in {target_language}
- Preserve all code snippets, variable names, and technical symbols exactly as written
- Maintain the same tone and formality level
- Do not add or remove information

Return ONLY the improved translation in {target_language}. Do not include any notes, explanations, or the original text.""",
            
            "llm_translation": """You are a professional translator specializing in technical content in {target_language}. Translate the following English text into {target_language}. Your translation must be accurate, natural-sounding, and preserve the exact meaning of the original text.

Important guidelines:
- Use the standard technical terminology in {target_language}
- Preserve all code, variable names, and technical symbols exactly as written
- Translate acronyms only if they have standard translations in {target_language}
- Maintain the same level of technical precision

Respond with ONLY the complete translation in {target_language}. Do not include any notes, explanations, or the original text.""",
            
            "safety_check_prompt": """You are a translation safety checker for technical content.

Check if this text has been properly translated or if a question has been answered instead of translated.

The text should be a straightforward translation from English to another language, not an attempt to answer a technical question posed in the English text.

Original text:
[Text 1]

Translation:
[Text 2]

If the translation is actually answering a question instead of translating it, respond with the single word "ISSUE". Otherwise, respond with "OK"."""
        }
        
        # Initialize prompts structure
        self.prompts = {
            "math": {
                "llm": math_llm_prompts,
                "hybrid": math_hybrid_prompts
            },
            "general": {
                "llm": general_llm_prompts,
                "hybrid": general_hybrid_prompts
            },
            "technical": {
                "llm": technical_llm_prompts,
                "hybrid": technical_hybrid_prompts
            }
        }
        
        # Save default prompts to files
        self._save_all_prompts()
    
    def _save_all_prompts(self):
        """Save all prompts to their respective JSON files."""
        try:
            for dataset_type, translators in self.prompts.items():
                dataset_dir = os.path.join(self.prompts_dir, dataset_type)
                os.makedirs(dataset_dir, exist_ok=True)
                
                for translator_type, prompts in translators.items():
                    file_path = os.path.join(dataset_dir, f"{translator_type}.json")
                    
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(prompts, f, ensure_ascii=False, indent=4)
                        logger.info(f"Saved prompts for {dataset_type}/{translator_type}")
                    except Exception as e:
                        logger.error(f"Failed to save prompts to {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error saving prompts: {e}")
    
    def get_prompts(self, dataset_type: str, translator_type: str) -> Dict[str, str]:
        """
        Get the prompts for a specific dataset and translator.
        
        Args:
            dataset_type: Type of dataset ('math', 'general', 'technical', etc.)
            translator_type: Type of translator ('llm', 'hybrid', etc.)
            
        Returns:
            Dict containing prompt templates
        """
        # If dataset_type doesn't exist, fall back to "general"
        if dataset_type not in self.prompts:
            logger.warning(f"No prompts found for dataset '{dataset_type}', falling back to 'general'")
            dataset_type = "general" if "general" in self.prompts else "math"
        
        # If translator_type doesn't exist for this dataset, fall back to "llm"
        if translator_type not in self.prompts[dataset_type]:
            logger.warning(f"No prompts found for translator '{translator_type}' in dataset '{dataset_type}', falling back to 'llm'")
            translator_type = "llm"
        
        return self.prompts[dataset_type][translator_type]
    
    def update_prompts(self, dataset_type: str, translator_type: str, prompts: Dict[str, str]):
        """
        Update prompts for a specific dataset and translator.
        
        Args:
            dataset_type: Type of dataset ('math', 'general', 'technical', etc.)
            translator_type: Type of translator ('llm', 'hybrid', etc.)
            prompts: Dictionary of prompt templates
        """
        if dataset_type not in self.prompts:
            self.prompts[dataset_type] = {}
        
        self.prompts[dataset_type][translator_type] = prompts
        
        # Save the updated prompts
        dataset_dir = os.path.join(self.prompts_dir, dataset_type)
        os.makedirs(dataset_dir, exist_ok=True)
        
        file_path = os.path.join(dataset_dir, f"{translator_type}.json")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, ensure_ascii=False, indent=4)
            logger.info(f"Updated prompts for {dataset_type}/{translator_type}")
        except Exception as e:
            logger.error(f"Failed to save updated prompts to {file_path}: {e}")