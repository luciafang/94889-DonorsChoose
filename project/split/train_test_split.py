import sklearn as sk
import pandas as pd
from imblearn.over_sampling import SMOTE
import json

def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def split_data(df, drop_year, test_year):
    '''
    df: pandas dataframe dataset of features and outcome variable
    drop_year: year of data that needs dropped
    test_year: year to use as test dataset
    '''
    # drop data without ground truth
    new_df = df[df['date_posted'].str[:4] != drop_year]
    test_df = new_df[new_df['date_posted'].str[:4] == test_year]

    # split off data used for testing (2013)
    train_df = new_df[new_df['date_posted'].str[:4] != test_year]
    
    # for projects posted in september, if not fully funded, drop
    train_df = train_df.drop(train_df[(train_df['date_posted'].str[5:7] == '09') & (train_df['not_fully_funded'] == 0)].index)

    # return training data and testing data sets
    return train_df.sort_values(by="date_posted"), test_df.sort_values(by="date_posted")

if __name__ == "__main__":
    cleaned_dataset_path = "../outputs/cleaned_dataset.csv"
    df = pd.read_csv(cleaned_dataset_path)
    df = df.drop('Unnamed: 0', axis=1) 

    config = load_config()

    drop_year = config["test_splits"]["drop"]
    test_year = config["test_splits"]["test"]

    train_df, test_df = split_data(df, drop_year, test_year)

    train_path = "../outputs/train_df.csv"
    test_path = "../outputs/test_df.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train/test split complete.")