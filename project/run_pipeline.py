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
    feature_dir = os.path.join(current_dir, "features")
    split_dir = os.path.join(current_dir, "split")
    model_dir = os.path.join(current_dir, "model")
    recommendation_dir = os.path.join(current_dir, "model")

    # Define paths for the scripts
    merge_script = os.path.join(preprocessing_dir, "feature_selection.py")
    clean_script = os.path.join(preprocessing_dir, "data_cleaning.py")
    feature_script = os.path.join(feature_dir, "feature_engineering.py")
    split_script = os.path.join(split_dir, "train_test_split.py")
    balance_script =os.path.join(preprocessing_dir, "data_segment_and_balance.py")
    model_script = os.path.join(model_dir, "model_training.py")
    eval_script = os.path.join(model_dir, "model_evaluation.py")
    importance_script = os.path.join(model_dir, "feature_importance.py")
    recommendation_script = os.path.join(recommendation_dir, "recommendation.py")

    ############################
    # Run the merge script
    print("Running dataset merging...")
    subprocess.run(["python", merge_script], cwd=preprocessing_dir, check=True)

    ############################
    # Run the feature script
    print("Running feature engineering...")
    subprocess.run(["python", feature_script], cwd=feature_dir, check=True)

    ############################
    # Run the cleaning script
    print("Running data cleaning...")
    subprocess.run(["python", clean_script], cwd=preprocessing_dir, check=True)

    ############################
    # run train/test and future data split
    print("Running train/test and future data split...")
    subprocess.run(["python", split_script], cwd=split_dir, check=True)

    ############################
    # run data balance and segment script
    print("Running data balancing and segmenting split...")
    subprocess.run(["python", balance_script], cwd=preprocessing_dir, check=True)

    ############################
    # run model training
    print("Running model training...")
    subprocess.run(["python", model_script], cwd=model_dir, check=True)

    # ############################
    # # run model evaluation
    print("Running model evaluation...")
    subprocess.run(["python", eval_script], cwd=model_dir, check=True)

    ############################
    # run feature importance
    print("Running feature importance...")
    subprocess.run(["python", importance_script], cwd=model_dir, check=True)

    ############################
    # Run recommendation generation
    print("Running recommendation generation...")
    subprocess.run(["python", recommendation_script], cwd=recommendation_dir, check=True)

    print("Pipeline complete.")
    ############################

if __name__ == "__main__":
    run_pipeline()