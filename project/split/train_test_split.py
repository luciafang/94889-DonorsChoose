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

    # split off data used for testing (2013)
    train_df = new_df[new_df['date_posted'].str[:4] != test_year]
    # for projects posted in september, if not fully funded, drop
    train_df = train_df.drop(train_df[(train_df['date_posted'].str[5:7] == '09') & (train_df['fully_funded'] == 0)].index)

    test_df = new_df[new_df['date_posted'].str[:4] == test_year]

    # return training data and testing data sets
    return train_df.sort_values(by="date_posted"), test_df.sort_values(by="date_posted")

def smote_balancing(df, pov_col_name, config):
    '''
    Use SMOTE to resample/upsample data to balance the two classes
    model_type: string name
    X_train: x training data (pandas dataframe)
    y_train: y training data (pandas dataframe)
    X_test: x testing data (pandas dataframe)
    '''
    smote = SMOTE(random_state=42)

    X = df.drop(['fully_funded', 'date_posted'], axis=1)
    y = df['fully_funded']

    new_X = X[X[pov_col_name] == True].drop(columns=[
        *config["poverty_columns"].values()], axis=1)

    new_y = y[X[pov_col_name] == True]

    X_res, y_res = smote.fit_resample(new_X, new_y)
    print(f'SMOTE upsampled from {len(new_y)} to {len(y_res)} samples')

    X_res["fully_funded"] = y_res
    return X_res

if __name__ == "__main__":
    cleaned_dataset_path = "../outputs/cleaned_dataset.csv"
    df = pd.read_csv(cleaned_dataset_path)

    config = load_config()

    drop_year = config["test_splits"]["drop"]
    test_year = config["test_splits"]["test"]

    train_df, test_df = split_data(df, drop_year, test_year)

    train_path = "../outputs/train_df.csv"
    test_path = "../outputs/test_df.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    for pov_lvl, pov_col_name in config["poverty_columns"].items():
        smote_dataset = smote_balancing(train_df, pov_col_name, config)
        smote_test_dataset = smote_balancing(test_df, pov_col_name, config)
        smote_path = f"../outputs/{pov_lvl}_pov_lvl_train_df.csv"
        smote_test_path = f"../outputs/{pov_lvl}_pov_lvl_test_df.csv"
        smote_dataset.to_csv(smote_path, index=False)
        smote_test_dataset.to_csv(smote_test_path, index=False)

    print(f"Train/test split complete.")