import re
import argparse

def parse_tag_content(text, tag):
    """Extract content from specified tag in the output
    
    Args:
        text (str): Text content to parse
        tag (str): Tag name to extract content from
    """
    pattern = fr"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        content = match.group(1).strip()
        return content
    else:
        return ""

def clean_text(text):
    """Clean hidden characters and irregular line breaks in text using regular expressions
    """
    # Remove zero-width space
    text = text.replace('\ufeff', '')
    # Normalize line endings
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
        print(f"## Error: {description} file '{file_path}' does not exist")
        raise
    except IOError as e:
        print(f"## Error: Cannot read {description} file '{file_path}': {str(e)}")
        raise

def ensure_all_tags_closed(text):
    """Ensure all opened tags have corresponding closing tags
    
    Args:
        text (str): Input text content
        
    Returns:
        str: Fixed text content
    """
    tags = re.findall(r'<(\w+)>', text)
    for tag in tags:
        if f"</{tag}>" not in text:
            text += f"</{tag}>"
    return text

def main():
    """Main program, combine all together"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract <report> section and print it to STD OUT")
    parser.add_argument('file_path', type=str, help='Path to the input file')
    args = parser.parse_args()

    # Read file content
    description = "input"
    text = read_file_safely(args.file_path, description)

    # Fix missing tags
    fixed_text = ensure_all_tags_closed(text)

    # Extract report tag content
    result_content = parse_tag_content(fixed_text, "report")

    # Clean result
    cleaned_result = clean_text(result_content)

    # Output result
    print(cleaned_result)

if __name__ == "__main__":
    main()
