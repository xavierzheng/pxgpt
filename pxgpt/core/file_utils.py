"""File operations and text processing utilities."""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """Clean hidden characters and irregular line breaks in text using regex
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove zero-width spaces
    text = text.replace('\ufeff', '')
    
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove consecutive line breaks
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text


def read_file_safely(file_path: str, description: str = "file") -> str:
    """Safely read file content
    
    Args:
        file_path: File path
        description: File description for error messages
        
    Returns:
        File content
        
    Raises:
        FileNotFoundError: File does not exist
        IOError: Error reading file
    """
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            content = f.read().strip()
            return clean_text(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: {description} file '{file_path}' does not exist")
    except IOError as e:
        raise IOError(f"Error: Cannot read {description} file '{file_path}': {str(e)}")


def write_file_safely(file_path: str, content: str, description: str = "file") -> None:
    """Safely write content to file
    
    Args:
        file_path: File path
        content: Content to write
        description: File description for error messages
        
    Raises:
        IOError: Error writing file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except IOError as e:
        raise IOError(f"Error: Cannot write to {description} file '{file_path}': {str(e)}")