import subprocess
import json
import os

def load_config(config_file="config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def run_pipeline():
    # Get the current directory of the run_pipeline script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the preprocessing directory
    preprocessing_dir = os.path.join(current_dir, "preprocessing")

    # Define paths for the scripts
    merge_script = os.path.join(preprocessing_dir, "feature_selection.py")
    clean_script = os.path.join(preprocessing_dir, "data_cleaning.py")

    ############################
    # Run the merge script
    print("Running dataset merging...")
    subprocess.run(["python", merge_script], cwd=preprocessing_dir, check=True)

    # Run the cleaning script
    print("Running data cleaning...")
    subprocess.run(["python", clean_script], cwd=preprocessing_dir, check=True)

    print("Pipeline complete.")
    ############################

if __name__ == "__main__":
    run_pipeline()