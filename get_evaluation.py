import os
import threading
import requests
import json

# Configuration
API_URL = "https://api.x.ai/v1"  # Change this if using a custom endpoint
API_KEY = open("C:/Users/berti/api_grok.txt", "r").read().strip()  # Replace with your actual API key
MODEL_NAME = "gpt-5"  # The model name for the answers folder, e.g., "gpt-4"
EVAL_MODEL = "grok-4-fast-non-reasoning"  # Fixed evaluation model
answers_dir = f"real_answers/{MODEL_NAME}"
gt_dir = "gt_answers"
eval_dir = f"evaluation/{MODEL_NAME}"

def process_file(filename):
    """
    Process a single answer file: compare with GT using API and save evaluation if not exists.
    """
    answer_path = os.path.join(answers_dir, filename)
    gt_path = os.path.join(gt_dir, filename)
    eval_path = os.path.join(eval_dir, filename)
    
    # Skip if evaluation already exists
    if os.path.exists(eval_path):
        print(f"Skipping {filename}: evaluation already exists.")
        return
    
    # Ensure evaluation directory exists
    os.makedirs(eval_dir, exist_ok=True)
    
    try:
        # Read the answer
        if not os.path.exists(answer_path):
            print(f"Skipping {filename}: answer file does not exist.")
            return
        with open(answer_path, 'r', encoding='utf-8') as f:
            answer = f.read().strip()
        
        # Read the ground-truth
        if not os.path.exists(gt_path):
            print(f"Skipping {filename}: ground-truth file does not exist.")
            return
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt = f.read().strip()
        
        if not answer or not gt:
            print(f"Skipping {filename}: empty answer or ground-truth.")
            return
        
        # Prepare the comparison prompt
        prompt = f"""Compare if the provided answer is equal (in the outcome, while the motivation steps can be different) to the ground-truth answer.
Provided answer:
{answer}

Ground-truth answer:
{gt}

Respond with only "yes" if the answer is corresponding (in the outcome), or "no" otherwise."""
        
        # Prepare the request
        endpoint = f"{API_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": EVAL_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,  # Short response expected
            "temperature": 0.0  # Deterministic for yes/no
        }
        
        # Send to API
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()
        
        response_data = response.json()
        eval_response = response_data["choices"][0]["message"]["content"].strip().lower()
        
        # Validate: must be "yes" or "no"
        if eval_response not in ["yes", "no"]:
            print(f"Invalid evaluation for {filename}: '{eval_response}'. Skipping.")
            return
        
        # Save the evaluation
        with open(eval_path, 'w', encoding='utf-8') as f:
            f.write(eval_response)
        
        print(f"Processed {filename}: saved evaluation '{eval_response}'.")
        
    except requests.exceptions.RequestException as e:
        print(f"Request error processing {filename}: {str(e)}")
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Response parsing error for {filename}: {str(e)}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

def main():
    """
    Main function: discover answer files and process in separate threads.
    Assumes answer files are .txt files in the 'answers_dir' directory.
    """
    if not os.path.exists(answers_dir):
        print(f"Directory '{answers_dir}' does not exist.")
        return
    
    if not os.path.exists(gt_dir):
        print(f"Directory '{gt_dir}' does not exist.")
        return
    
    # Get list of answer files (assuming .txt extension)
    answer_files = [f for f in os.listdir(answers_dir) if f.endswith('.txt')]
    
    if not answer_files:
        print(f"No .txt files found in '{answers_dir}' directory.")
        return
    
    print(f"Found {len(answer_files)} answer files. Starting threaded evaluation...")
    
    # Create and start threads for each file
    threads = []
    for filename in answer_files:
        thread = threading.Thread(target=process_file, args=(filename,))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("All evaluations completed.")

if __name__ == "__main__":
    main()