import os
import pandas as pd

# Configuration
PROMPTS_DIR = "prompts"
OUTPUT_DIR = "prompts_csv"

def process_prompt(filename):
    """
    Process a single prompt file: extract CSV lines from the bottom and remainder.
    """
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    
    if not os.path.exists(prompt_path):
        print(f"Skipping {filename}: file does not exist.")
        return
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        print(f"Skipping {filename}: empty file.")
        return
    
    # Collect CSV lines from the bottom: lines with more than 2 commas
    csv_lines = []
    for line in reversed(lines):
        if line.count(',') > 2:
            csv_lines.append(line.rstrip('\n') + '\n')  # Ensure consistent line endings
        else:
            break
    
    if not csv_lines:
        print(f"No CSV lines found in {filename}.")
        # Still create remainder as full content
        csv_lines = []
    
    csv_lines.reverse()  # Restore original order
    csv_content = ''.join(csv_lines).rstrip()  # Remove trailing newline if any
    
    # Create subdirectory
    subdir_name = filename[:-4]  # Remove .txt extension
    subdir = os.path.join(OUTPUT_DIR, subdir_name)
    os.makedirs(subdir, exist_ok=True)
    
    # Write CSV
    csv_path = os.path.join(subdir, "table.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)
    
    # Validate CSV with pandas
    try:
        df = pd.read_csv(csv_path)
        print(f"Validated CSV for {filename}: {df.shape}")
    except Exception as e:
        # Remove invalid CSV file
        if os.path.exists(csv_path):
            os.remove(csv_path)
        raise Exception(f"CSV for {filename} is invalid and cannot be imported with pd.read_csv: {str(e)}")
    
    # Write remainder (lines before CSV part)
    num_csv_lines = len(csv_lines)
    remainder_lines = lines[:len(lines) - num_csv_lines]
    remainder_content = ''.join(remainder_lines).rstrip()
    
    remainder_path = os.path.join(subdir, "remainder.txt")
    with open(remainder_path, 'w', encoding='utf-8') as f:
        f.write(remainder_content)
    
    print(f"Processed {filename}: CSV and remainder saved.")

def main():
    """
    Main function: process all .txt files in the prompts directory.
    """
    if not os.path.exists(PROMPTS_DIR):
        print(f"Directory '{PROMPTS_DIR}' does not exist.")
        return
    
    # Get list of prompt files
    prompt_files = [f for f in os.listdir(PROMPTS_DIR) if f.endswith('.txt')]
    
    if not prompt_files:
        print(f"No .txt files found in '{PROMPTS_DIR}' directory.")
        return
    
    print(f"Found {len(prompt_files)} prompt files. Starting processing...")
    
    for filename in prompt_files:
        try:
            process_prompt(filename)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            # Optionally continue or raise to stop
    
    print("All processing completed.")

if __name__ == "__main__":
    main()