import os
import threading
import requests
import json

# Configuration
API_URL = "https://api.x.ai/v1"  # Change this if using a custom endpoint
API_KEY = open("C:/Users/berti/api_grok.txt", "r").read().strip()  # Replace with your actual API key
MODEL1 = "gpt-5-csv-agent"  # First model for comparison
MODEL2 = "gpt-5-mini-csv-agent"  # Second model for comparison
EVAL_MODEL = "grok-4-fast-non-reasoning"  # Fixed evaluation model
answers_dir1 = f"real_answers/{MODEL1}"
answers_dir2 = f"real_answers/{MODEL2}"
eval_dir = f"cross_evaluation/{MODEL1}_vs_{MODEL2}"

def process_file(filename):
    """
    Process a single answer file pair: compare MODEL1 with MODEL2 using API and save evaluation if not exists.
    """
    answer_path1 = os.path.join(answers_dir1, filename)
    answer_path2 = os.path.join(answers_dir2, filename)
    eval_path = os.path.join(eval_dir, filename)
    
    # Skip if evaluation already exists
    if os.path.exists(eval_path):
        print(f"Skipping {filename}: evaluation already exists.")
        return
    
    # Ensure evaluation directory exists
    os.makedirs(eval_dir, exist_ok=True)
    
    try:
        # Read the answer from MODEL1
        if not os.path.exists(answer_path1):
            print(f"Skipping {filename}: answer file for {MODEL1} does not exist.")
            return
        with open(answer_path1, 'r', encoding='utf-8') as f:
            answer1 = f.read().strip()
        
        # Read the answer from MODEL2 (as reference)
        if not os.path.exists(answer_path2):
            print(f"Skipping {filename}: answer file for {MODEL2} does not exist.")
            return
        with open(answer_path2, 'r', encoding='utf-8') as f:
            answer2 = f.read().strip()
        
        if not answer1 or not answer2:
            print(f"Skipping {filename}: empty answer for {MODEL1} or {MODEL2}.")
            return
        
        # Prepare the comparison prompt
        prompt = f"""Compare if the provided answer is equal to the reference answer.
Provided answer ({MODEL1}):
{answer1}

Reference answer ({MODEL2}):
{answer2}

Respond with only "yes" if the answer is corresponding, or "no" otherwise."""
        
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
    Main function: discover answer files from MODEL1 and process pairs in separate threads.
    Assumes answer files are .txt files in the 'answers_dir1' directory.
    """
    if not os.path.exists(answers_dir1):
        print(f"Directory '{answers_dir1}' does not exist.")
        return
    
    if not os.path.exists(answers_dir2):
        print(f"Directory '{answers_dir2}' does not exist.")
        return
    
    # Get list of answer files from MODEL1 (assuming .txt extension)
    answer_files = [f for f in os.listdir(answers_dir1) if f.endswith('.txt')]
    
    if not answer_files:
        print(f"No .txt files found in '{answers_dir1}' directory.")
        return
    
    print(f"Found {len(answer_files)} answer files. Starting threaded cross-evaluation...")
    
    # Create and start threads for each file
    threads = []
    for filename in answer_files:
        thread = threading.Thread(target=process_file, args=(filename,))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("All cross-evaluations completed.")

if __name__ == "__main__":
    main()