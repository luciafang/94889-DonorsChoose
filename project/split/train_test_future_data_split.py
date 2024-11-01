import sklearn as sk
import pandas as pd
from imblearn.over_sampling import SMOTE
import json

def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def split_data(df):
    '''
    df: pandas dataframe dataset of features and outcome variable
    '''
    # Split the data into features (X) and target (y)
    # for training and testing, we focus on projects date_posted before 2014-01-12
    # rest is treated as future data (2014-01-12->2014-05-11, which is max date)
    df['date_posted'] = pd.to_datetime(df['date_posted'], format='%Y-%m-%d', errors='coerce')
    max_date = df['date_posted'].max()
    cutoff_date = max_date - pd.DateOffset(months=4)
    train_test_df = df[df['date_posted'] <= cutoff_date].sort_values(by=['date_posted']).copy().drop('date_posted', axis=1)
    future_df = df[df['date_posted'] > cutoff_date].sort_values(by=['date_posted']).copy().drop('date_posted', axis=1)

    # also there may be projects that didn't finish their funding rounds (-1) dropping the unknowns
    filtered_train_test_df = train_test_df[train_test_df['fully_funded'] != -1]

    return filtered_train_test_df, future_df 

def smote_balancing(df, pov_col_name, config):
    '''
    Use SMOTE to resample/upsample data to balance the two classes
    model_type: string name
    X_train: x training data (pandas dataframe)
    y_train: y training data (pandas dataframe)
    X_test: x testing data (pandas dataframe)
    '''
    smote = SMOTE(random_state=42)

    X = df.drop(['fully_funded', 'projectid', 'date_posted'], axis=1)
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

    train_test_df, future_df = split_data(df)

    train_test_path = "../outputs/train_test_df.csv"
    future_path = "../outputs/future_df.csv"
    
    train_test_df.to_csv(train_test_path, index=False)
    future_df.to_csv(future_path, index=False)

    config = load_config()

    for pov_lvl, pov_col_name in config["poverty_columns"].items():
        smote_dataset = smote_balancing(df, pov_col_name, config)
        smote_path = f"../outputs/{pov_lvl}_pov_lvl_train_test_df.csv"
        smote_dataset.to_csv(smote_path, index=False)

    print(f"Train/test and future data split complete.")