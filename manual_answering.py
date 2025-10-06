import os
import pyperclip

# Hard-coded model name
MODEL_NAME = "gpt-5-csv-agent"  # Change this to the desired model name

# Directories
PROMPTS_CSV_DIR = "prompts_csv"
ANSWERS_DIR = f"real_answers/{MODEL_NAME}"

def main():
    """
    Main function: interactively process subdirectories in prompts_csv.
    Copies remainder.txt to clipboard, prints subdir name, opens answer file for manual edit.
    """
    # Ensure answers directory exists
    os.makedirs(ANSWERS_DIR, exist_ok=True)
    
    if not os.path.exists(PROMPTS_CSV_DIR):
        print(f"Directory '{PROMPTS_CSV_DIR}' does not exist.")
        return
    
    # Get list of subdirectories
    subdirs = [d for d in os.listdir(PROMPTS_CSV_DIR) 
               if os.path.isdir(os.path.join(PROMPTS_CSV_DIR, d))]
    
    if not subdirs:
        print(f"No subdirectories found in '{PROMPTS_CSV_DIR}'.")
        return
    
    print(f"Found {len(subdirs)} subdirectories. Starting manual processing...")
    
    for subdir in subdirs:
        answer_filename = f"{subdir}.txt"
        answer_path = os.path.join(ANSWERS_DIR, answer_filename)
        
        # Skip if answer already exists
        if os.path.exists(answer_path):
            print(f"Skipping {subdir}: answer already exists.")
            continue
        
        remainder_path = os.path.join(PROMPTS_CSV_DIR, subdir, "remainder.txt")
        
        # Check if remainder exists
        if not os.path.exists(remainder_path):
            print(f"Skipping {subdir}: remainder.txt does not exist.")
            continue
        
        try:
            # Read and copy prompt to clipboard
            with open(remainder_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            pyperclip.copy(prompt)
            
            print(f"\n--- Processing: {subdir} ---")
            print("Prompt copied to clipboard. Paste into LLM and copy response back.")
            
            # Create empty answer file
            with open(answer_path, 'w', encoding='utf-8') as f:
                pass  # Empty file
            
            # Open the answer file with default editor (assuming Windows with notepad)
            # For cross-platform, adjust as needed (e.g., 'open' on macOS, 'xdg-open' on Linux)
            os.system(f"notepad.exe \"{answer_path}\"")
            
            # Wait for user to finish pasting and saving
            input("Press Enter after saving the LLM response in the file...")
            
        except Exception as e:
            print(f"Error processing {subdir}: {str(e)}")
            continue
    
    print("All manual processing completed.")

if __name__ == "__main__":
    main()