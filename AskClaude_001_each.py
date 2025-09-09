import sys
import time
from pathlib import Path
import base64
from anthropic import Anthropic, RateLimitError, APIConnectionError, APIStatusError

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_multi_image_message(folder_path, prompt_text):
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

def process_with_rate_limit(client, folder_path, system_promt, prompt_text):
    try:
        messages = create_multi_image_message(folder_path, prompt_text)
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219", # model claude-3-5-sonnet-20241022 claude-3-7-sonnet-20250219
            max_tokens=8192,
            temperature=0.5,
            system=system_promt,
            messages=messages
        )
        # Print request ID
        print(f"Request ID: {response._request_id}")
        return response
    # for debug
    except RateLimitError:
        time.sleep(60)
        return process_with_rate_limit(client, folder_path, system_promt, prompt_text)
    except APIConnectionError as e:
        print(f"Connection error: {e.__cause__}")
        raise
    except APIStatusError as e:
        print(f"API error: Status {e.status_code}")
        print(f"Response: {e.response}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python AskClaude.py <input_folder> <output_file> <system_prompt_file> <prompt_file>")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    output_file = sys.argv[2]
    system_prompt_file = sys.argv[3]
    prompt_file = sys.argv[4]

    # Read system prompt and prompt from files
    with open(system_prompt_file, "r") as spf:
        my_system_prompt = spf.read().strip()

    with open(prompt_file, "r") as pf:
        my_prompt = pf.read().strip()
    
    # Create link. Default try 2 times if any error
    client = Anthropic(max_retries=2)
    response = process_with_rate_limit(client, input_folder, my_system_prompt, my_prompt)
    
    with open(output_file, 'w') as f:
        f.write(response.content[0].text)
