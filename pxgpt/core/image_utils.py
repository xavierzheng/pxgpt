"""Image processing utilities."""

import base64
from pathlib import Path
from typing import List, Dict, Any


def get_base64_encoded_image(image_path: str) -> str:
    """Convert image to base64 encoding
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_image_content_list(folder_path: str) -> List[Dict[str, Any]]:
    """Convert all .jpg files in folder to base64 content list
    
    Args:
        folder_path: Path to folder containing images
        
    Returns:
        List of image content dictionaries for API
    """
    image_paths = list(Path(folder_path).glob("*.jpg"))
    content_list = []
    
    for img_path in image_paths:
        content_list.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": get_base64_encoded_image(str(img_path))
            }
        })
    
    return content_list


def create_multi_image_message(folder_path: str, prompt_text: str) -> List[Dict[str, Any]]:
    """Create message with multiple images and text prompt
    
    Args:
        folder_path: Path to folder containing images
        prompt_text: Text prompt
        
    Returns:
        Message list for API with images and prompt, images first
    """
    content_list = create_image_content_list(folder_path)
    content_list.append({"type": "text", "text": prompt_text})
    
    return [{"role": "user", "content": content_list}]