import sklearn as sk
import pandas as pd
from imblearn.over_sampling import SMOTE
import json

def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

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
    config = load_config()

    train_path = "../outputs/train_df.csv"
    test_path = "../outputs/test_df.csv"

    train_df = pd.read_csv(train_path)
    train_df = train_df.set_index("projectid")
    test_df = pd.read_csv(test_path)
    test_df = test_df.set_index("projectid")

    for pov_lvl, pov_col_name in config["poverty_columns"].items():
        smote_dataset = smote_balancing(train_df, pov_col_name, config)
        smote_test_dataset = smote_balancing(test_df, pov_col_name, config)
        smote_path = f"../outputs/{pov_lvl}_pov_lvl_train_df.csv"
        smote_test_path = f"../outputs/{pov_lvl}_pov_lvl_test_df.csv"
        smote_dataset.to_csv(smote_path, index=False)
        smote_test_dataset.to_csv(smote_test_path, index=False)

    print(f"Data balancing and segmenting complete.")