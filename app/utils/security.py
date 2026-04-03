"""
Security utilities for input validation and prompt injection defense.
"""
import re
from typing import List
from app.utils.logger import get_logger

logger = get_logger(__name__)

# List of common prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"system prompt",
    r"you are now (a|an)",
    r"new instructions",
    r"disregard",
    r"bypass",
    r"reveal your secrets",
    r"output in a different format",
]

class SecurityGuard:
    """Validate and sanitize user input."""
    
    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """Sanitize prompt by removing potentially dangerous characters."""
        # Remove null bytes
        prompt = prompt.replace('\x00', '')
        return prompt.strip()

    @staticmethod
    def detect_injection(prompt: str) -> bool:
        """Detect potential prompt injection attacks."""
        prompt_lower = prompt.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, prompt_lower):
                logger.warning(f"Potential injection detected: '{pattern}' matched.")
                return True
        return False

    @staticmethod
    def is_valid_prompt(prompt: str, min_length: int = 1, max_length: int = 4000) -> bool:
        """Check if prompt meets length constraints."""
        length = len(prompt)
        if length < min_length:
            return False
        if length > max_length:
            return False
        return True

    @staticmethod
    def validate_file_type(filename: str, allowed_extensions: List[str]) -> bool:
        """Validate uploaded file extension."""
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        return ext in allowed_extensions
