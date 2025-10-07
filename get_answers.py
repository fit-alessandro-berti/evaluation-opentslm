import os
import threading
import requests
import json

# Configuration
API_URL = "https://openrouter.ai/api/v1"  # Change this if using a custom endpoint
API_KEY = open("C:/Users/berti/api_openrouter.txt", "r").read().strip()  # Replace with your actual API key
MODEL_NAME = "qwen/qwen3-235b-a22b-thinking-2507"  # Change to desired model, e.g., "gpt-4"
answers_dir = "real_answers/" + MODEL_NAME

def process_prompt(filename):
    """
    Process a single prompt file: send to API and save response if no answer exists.
    """
    prompt_dir = "prompts"
    
    prompt_path = os.path.join(prompt_dir, filename)
    answer_path = os.path.join(answers_dir, filename)
    
    # Skip if answer already exists
    if os.path.exists(answer_path):
        print(f"Skipping {filename}: answer already exists.")
        return
    
    # Ensure answers directory exists
    os.makedirs(answers_dir, exist_ok=True)
    
    try:
        # Read the prompt
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        if not prompt:
            print(f"Skipping {filename}: empty prompt.")
            return
        
        # Prepare the request
        endpoint = f"{API_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # Send to API
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()
        
        response_data = response.json()
        answer = response_data["choices"][0]["message"]["content"].strip()
        
        # Save the answer
        with open(answer_path, 'w', encoding='utf-8') as f:
            f.write(answer)
        
        print(f"Processed {filename}: saved answer.")
        
    except requests.exceptions.RequestException as e:
        print(f"Request error processing {filename}: {str(e)}")
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Response parsing error for {filename}: {str(e)}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

def main():
    """
    Main function: discover prompt files and process in separate threads.
    Assumes prompt files are .txt files in the 'prompts' directory.
    """
    prompt_dir = "prompts"
    
    if not os.path.exists(prompt_dir):
        print(f"Directory '{prompt_dir}' does not exist. Creating it.")
        os.makedirs(prompt_dir, exist_ok=True)
        return
    
    # Get list of prompt files (assuming .txt extension)
    prompt_files = [f for f in os.listdir(prompt_dir) if f.endswith('.txt')]
    
    if not prompt_files:
        print("No .txt files found in 'prompts' directory.")
        return
    
    print(f"Found {len(prompt_files)} prompt files. Starting threaded processing...")
    
    # Create and start threads for each file
    threads = []
    for filename in prompt_files:
        thread = threading.Thread(target=process_prompt, args=(filename,))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("All processing completed.")

if __name__ == "__main__":
    main()