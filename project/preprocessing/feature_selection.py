import pandas as pd
import json
import os

def select_features(df, features):
    """
    Select specified features from the DataFrame.
    
    :param df: DataFrame to select features from
    :param features: List of features to select
    :return: DataFrame with only the selected features
    """
    # Ensure 'projectid' is always included in the features
    if 'projectid' not in features:
        features.insert(0, 'projectid')
    
    # Select only the requested features
    selected_df = df[features] if all(feature in df.columns for feature in features) else df
    return selected_df

def merge_datasets(donations_file, essays_file, projects_file, resources_file, outcomes_file):
    """
    Merge datasets based on project ID and return the merged DataFrame.
    
    :param donations_file: Path to the donations CSV file
    :param essays_file: Path to the essays CSV file
    :param projects_file: Path to the projects CSV file
    :param resources_file: Path to the resources CSV file
    :param outcomes_file: Path to the outcomes CSV file
    :return: Merged DataFrame
    """
    # donations = pd.read_csv(donations_file)
    # essays = pd.read_csv(essays_file)
    projects = pd.read_csv(projects_file, parse_dates=['date_posted'])

    # move somewhere
    # # find max date (05-11-2014) and remove any projects that was posted after (02-11-2014)
    # # keeping projects that have been there for 3+ months
    # max_date = projects['date_posted'].max()
    # cutoff_date = max_date - pd.DateOffset(months=3)
    # projects_not_recent = projects[projects['date_posted'] <= cutoff_date]
    # print('Number of projects not recent: {}'.format(len(projects_not_recent)))
    # resources = pd.read_csv(resources_file)
    outcomes = pd.read_csv(outcomes_file)

    merged_data = pd.merge(projects, outcomes, on='projectid', how='left')
    # merged_data = pd.merge(merged_data, essays, on=['projectid', 'teacher_acctid'], how='left')

    return merged_data


def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    config = load_config()
    
    raw_datasets = config["raw_datasets"]

    merged_data = merge_datasets(
        raw_datasets["donations"],
        raw_datasets["essays"],
        raw_datasets["projects"],
        raw_datasets["resources"],
        raw_datasets["outcomes"]
    )

    # Select features from the merged dataset
    features_to_use = config.get("features_to_use", [])
    df_selected = select_features(merged_data, features_to_use)
    df_selected['fully_funded'] = df_selected['fully_funded'].map({'t': int(1), 'f': int(0)})
    dataset_path = "../outputs/selected_dataset.csv"

    output_dir = os.path.dirname(dataset_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the selected features to a new CSV
    df_selected.to_csv(dataset_path, index=False)
    print(f"Selected features saved as '{dataset_path}'.")