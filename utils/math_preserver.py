"""
Utility to preserve mathematical expressions during translation.
"""

import re
import uuid
from typing import Dict, Tuple, List
from utils.logger import logger

class SimpleMathPreserver:
    """
    Utility class to extract, preserve, and restore mathematical expressions during translation.
    Uses a combination of regex patterns to identify and protect various math notation formats.
    """
    
    def __init__(self):
        """Initialize the math preserver with regex patterns for different math notations."""
        
        # Define regex patterns for different types of math expressions
        self.patterns = [
            # LaTeX inline math - $...$
            r'\$[^$]+\$',
            
            # LaTeX display math - $$...$$
            r'\$\$[^$]+\$\$',
            
            # LaTeX environments - \begin{...}...\end{...}
            r'\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}',
            
            # LaTeX commands with arguments - \command{...}
            r'\\[a-zA-Z]+\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            
            # Standalone LaTeX commands - \command
            r'\\[a-zA-Z]+',
            
            # Simple mathematical expressions - x+y=z, 2*3, etc.
            r'(?<!\w)(\d+[\+\-\*\/\^][\(\)\d\+\-\*\/\^\s]+\d+)(?!\w)',
            
            # Equations with variables - x = 2, y + 3 = 7, etc.
            r'(?<!\w)([a-zA-Z][a-zA-Z0-9_]*\s*[\+\-\*\/\^=<>]\s*[\(\)\d\+\-\*\/\^\s]+)(?!\w)',
            
            # Variable ranges - x_1, y^2, etc.
            r'[a-zA-Z]_\d+|[a-zA-Z]\^\d+',
            
            # Variable with subscripts - a_{min}, etc.
            r'[a-zA-Z]_\{[^}]+\}',
            
            # Equations wrapped in brackets - (x+y=z)
            r'\([a-zA-Z0-9\+\-\*\/\^\s=]+\)'
        ]
        
        logger.info("SimpleMathPreserver initialized")
    
    def extract_math(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract mathematical expressions from text and replace them with unique placeholders.
        
        Args:
            text: Input text that may contain mathematical expressions
            
        Returns:
            Tuple containing:
                - Modified text with placeholders
                - Dictionary mapping placeholders to original expressions
        """
        if not text:
            return text, {}
            
        replacements = {}
        modified_text = text
        
        try:
            # Process each pattern and replace matches with placeholders
            for pattern in self.patterns:
                matches = re.finditer(pattern, modified_text)
                
                # Create a list to store matches and their positions
                match_data = []
                
                # Collect all matches first
                for match in matches:
                    match_text = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()
                    match_data.append((match_text, start_pos, end_pos))
                
                # Process matches in reverse order (to avoid position shifts)
                for match_text, start_pos, end_pos in sorted(match_data, key=lambda x: x[1], reverse=True):
                    placeholder = f"__MATH_{uuid.uuid4().hex}__"
                    replacements[placeholder] = match_text
                    
                    # Replace the match with the placeholder
                    modified_text = modified_text[:start_pos] + placeholder + modified_text[end_pos:]
            
            logger.info(f"Extracted {len(replacements)} math expressions from text")
            
            return modified_text, replacements
            
        except Exception as e:
            logger.error(f"Error extracting math expressions: {e}")
            # Return original text if extraction fails
            return text, {}
    
    def restore_math(self, translated_text: str, replacements: Dict[str, str]) -> str:
        """
        Restore mathematical expressions in translated text by replacing placeholders.
        
        Args:
            translated_text: Translated text with placeholders
            replacements: Dictionary mapping placeholders to original expressions
            
        Returns:
            str: Text with mathematical expressions restored
        """
        if not translated_text or not replacements:
            return translated_text
            
        result = translated_text
        
        try:
            # Sort replacements by placeholder length (longest first)
            # This helps avoid partial replacement issues
            sorted_placeholders = sorted(replacements.keys(), key=len, reverse=True)
            
            for placeholder in sorted_placeholders:
                result = result.replace(placeholder, replacements[placeholder])
                
            logger.info(f"Restored {len(replacements)} math expressions in translated text")
            
            return result
            
        except Exception as e:
            logger.error(f"Error restoring math expressions: {e}")
            # Return translated text if restoration fails
            return translated_text