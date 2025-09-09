import sys
import time
import re
import argparse
from pathlib import Path
import base64
from anthropic import Anthropic, RateLimitError, APIConnectionError, APIStatusError

def get_base64_encoded_image(image_path):
    """Convert image to base64 encoding

    Args:
        image_path (string): Path to the image file

    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_multi_image_message(folder_path, prompt_text):
    """Convert all .jpg files in folder to base64 encoding

    Args:
        folder_path (string): Path to folder containing images
        prompt_text (string): Text prompt
        
    Returns:
        list: Message for Anthropic API containing images and prompt, with images first
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
    content_list.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content_list}]

def clean_text(text):
    """Clean hidden characters and irregular line breaks in text using regex
    """
    # Remove zero-width spaces
    text = text.replace('\ufeff', '')
    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove consecutive line breaks
    text = re.sub(r'\n\s*\n', '\n', text)
    return text

def read_file_safely(file_path, description):
    """Safely read file content
    
    Args:
        file_path (str): File path
        description (str): File description for error messages
        
    Returns:
        str: File content
        
    Raises:
        FileNotFoundError: File does not exist
        IOError: Error reading file
    """
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: {description} file '{file_path}' does not exist")
        raise
    except IOError as e:
        print(f"Error: Cannot read {description} file '{file_path}': {str(e)}")
        raise

def process_with_rate_limit(client, folder_path, system_prompt, json_single_line, prompt_text):
    """Process API request with retry mechanism
    
    Args:
        client (Anthropic): Anthropic API client
        folder_path (str): Path to folder containing images
        system_prompt (str): System prompt
        json_single_line (str): JSON file content
        prompt_text (str): User prompt
        
    Returns:
        Response: API response
    """
    try:
        messages = create_multi_image_message(folder_path, prompt_text)
        model_name="claude-3-7-sonnet-20250219" #claude-3-5-sonnet-20241022
        
        # Estimate token count
        count = client.beta.messages.count_tokens(
            model=model_name,
            messages=messages
        )
        print(f"## Estimated input tokens: {count.input_tokens}")
        
        # Send request
        response = client.messages.create(
            model=model_name,
            max_tokens=8192, # this is maximum
            temperature=0.5,
            system=[
                {
                    "type": "text",
                    "text": system_prompt
                },
                {
                    "type": "text",
                    "text": json_single_line,
                    "cache_control": {"type": "ephemeral"}
                }
                ],
            messages=messages
        )
        
        # Print request ID and actual usage, remember to use single quotes '' inside double quotes "" to avoid errors
        print(f"## Request ID: {response._request_id}")
        print(f"## cache_creation_input_tokens: {getattr(response.usage, 'cache_creation_input_tokens', 'NA')}")
        print(f"## cache_read_input_tokens: {getattr(response.usage, 'cache_read_input_tokens', 'NA')}")
        print(f"## Actual input tokens: {getattr(response.usage, 'input_tokens', 'NA')}")
        print(f"## Actual output tokens: {getattr(response.usage, 'output_tokens', 'NA')}")
        
        return response
    
    except RateLimitError:
        time.sleep(60)
        return process_with_rate_limit(client, folder_path, system_prompt, json_single_line, prompt_text)
    except APIConnectionError as e:
        print(f"Connection error: {e.__cause__}")
        raise
    except APIStatusError as e:
        print(f"API error: Status {e.status_code}")
        print(f"Response: {e.response}")
        raise

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate JSON files for plant images using Anthropic API")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to images")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path")
    parser.add_argument("--system_prompt", type=str, required=True, help="System prompt file")
    parser.add_argument("--json_single_line", type=str, required=True, help="JSON schema file")
    parser.add_argument("--prompt_text", type=str, required=True, help="User prompt file")
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Read files
    system_prompt_temp = read_file_safely(args.system_prompt, "system prompt") # Read file
    prompt_text_temp = read_file_safely(args.prompt_text, "prompt text") # Read file
    json_single_line_temp = read_file_safely(args.json_single_line, "json_single_line") # Read file
    system_prompt = clean_text(system_prompt_temp) # Clean text
    prompt_text = clean_text(prompt_text_temp) # Clean text
    json_single_line = clean_text(json_single_line_temp) # Clean text
    
    # Process API request
    client = Anthropic()
    response = process_with_rate_limit(client, args.folder_path, system_prompt, json_single_line, prompt_text)
    
    # Write output file
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(response.content[0].text)
        print(f"Results successfully written to file: {args.output_file}")
    except IOError as e:
        print(f"Error: Cannot write to output file '{args.output_file}': {str(e)}")
        raise

if __name__ == "__main__":
    main()
