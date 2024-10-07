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
    split_dir = os.path.join(current_dir, "split")
    model_dir = os.path.join(current_dir, "model")

    # Define paths for the scripts
    merge_script = os.path.join(preprocessing_dir, "feature_selection.py")
    clean_script = os.path.join(preprocessing_dir, "data_cleaning.py")
    split_script = os.path.join(split_dir, "data_train_test_split.py")
    model_script = os.path.join(model_dir, "model_training.py")
    eval_script = os.path.join(model_dir, "model_evaluation.py")

    ############################
    # Run the merge script
    print("Running dataset merging...")
    subprocess.run(["python", merge_script], cwd=preprocessing_dir, check=True)

    # Run the cleaning script
    print("Running data cleaning...")
    subprocess.run(["python", clean_script], cwd=preprocessing_dir, check=True)

    ############################
    # run train test split
    print("Running train test split...")
    subprocess.run(["python", split_script], cwd=split_dir, check=True)

    ############################
    # run model training
    print("Running model training...")
    subprocess.run(["python", model_script], cwd=model_dir, check=True)

    ############################
    # run model evaluation
    print("Running model evaluation...")
    subprocess.run(["python", eval_script], cwd=model_dir, check=True)

    print("Pipeline complete.")
    ############################

if __name__ == "__main__":
    run_pipeline()