import os

# Configuration
EVAL_DIR = "cross_evaluation/gpt-5_vs_gpt-5-mini"  # Change to the desired evaluation folder

def main():
    """
    Main function: count the percentage of 'yes' evaluations in the folder.
    Assumes evaluation files are .txt files containing 'yes' or 'no'.
    """
    if not os.path.exists(EVAL_DIR):
        print(f"Directory '{EVAL_DIR}' does not exist.")
        return
    
    # Get list of evaluation files (assuming .txt extension)
    eval_files = [f for f in os.listdir(EVAL_DIR) if f.endswith('.txt')]
    
    if not eval_files:
        print(f"No .txt files found in '{EVAL_DIR}' directory.")
        return
    
    yes_count = 0
    total_files = len(eval_files)
    
    for filename in eval_files:
        eval_path = os.path.join(EVAL_DIR, filename)
        try:
            with open(eval_path, 'r', encoding='utf-8') as f:
                content = f.read().strip().lower()
                if content == "yes":
                    yes_count += 1
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            continue
    
    if total_files > 0:
        percentage = (yes_count / total_files) * 100
        print(f"Total files: {total_files}")
        print(f"Yes count: {yes_count}")
        print(f"Percentage of 'yes': {percentage:.2f}%")
    else:
        print("No valid files to evaluate.")

if __name__ == "__main__":
    main()