import sklearn as sk
import pandas as pd
from imblearn.over_sampling import SMOTE
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def smote_balancing(df, pov_col_name, config):
    '''
    Use SMOTE to resample/upsample data to balance the two classes (not_fully_funded)
    so there is an even number for the specified poverty level

    df: the dataframe containing the features and outcome label
    pov_col_name: the name of the poverty level section of the dataset being balanced
    config: the loaded config file
    '''
    # initialize smote
    smote = SMOTE(random_state=42, k_neighbors=5)

    # sort dataset by date posted
    df['date_posted'] = pd.to_datetime(df['date_posted'])
    df = df.sort_values("date_posted")
    date_posted_col = df["date_posted"]

    # create X and y
    X = df.drop(['not_fully_funded', 'date_posted'], axis=1)
    y = df['not_fully_funded']

    # get only the rows with the specified poverty level
    new_X = X[X[pov_col_name] == True].drop(columns=[
        *config["poverty_columns"].values()], axis=1)

    new_y = y[X[pov_col_name] == True]

    # resample with smote to balance
    X_res, y_res = smote.fit_resample(new_X, new_y)

    # add interpolated dates to synthetic data so we can use TimeSeriesSplit for training
    original_sample_count = X.shape[0]
    balanced_sample_count = X_res.shape[0] - X.shape[0]

    # Combine original dates with indices
    original_dates = date_posted_col[X.index].reset_index(drop=True)
    synthetic_dates = []

    X_res_np = X_res.to_numpy()

    # Interpolate dates for synthetic samples
    for i in range(balanced_sample_count):
        # Find two nearest neighbors in the original data
        neighbor_indices = smote.nn_k_.kneighbors(
            X_res_np[original_sample_count + i].reshape(1, -1), 
            return_distance=False
        )[0][:2]

        neighbor_dates = original_dates.iloc[neighbor_indices].sort_values().values
        # Interpolate the date between the two neighbors
        interpolated_date = neighbor_dates[0] + (neighbor_dates[1] - neighbor_dates[0]) / 2
        synthetic_dates.append(interpolated_date)
    
    # Combine original and synthetic dates
    final_dates = pd.concat([
        original_dates,
        pd.Series(synthetic_dates, name='date_posted')
    ]).sort_index(ignore_index=True)

    print(f'SMOTE upsampled from {len(new_y)} to {len(y_res)} samples')

    # assign new dates and new balanced outcome variable
    X_res["date_posted"] = final_dates
    X_res["not_fully_funded"] = y_res
    return X_res.sort_values('date_posted')

def scale_quant_vars(df, quant_variables, pov_lvl="none"):
    """
    scale the quantitative variables with StandardScaler from sklearn

    df: dataset
    categorical_features: list of categorical features to exclude from scaling
    """
    quant_variables = df[quant_variables]
    categorical_vars = df.drop(quant_variables, axis=1)

    quant_features = quant_variables.columns

    scaler = MinMaxScaler()
    scaled_quant = scaler.fit_transform(quant_variables)

    scaler_path = f"../outputs/{pov_lvl}_poverty_level_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    df_scaled_quant = pd.DataFrame(scaled_quant, columns=quant_features, index=df.index)
    df_transformed = pd.concat([df_scaled_quant, categorical_vars], axis=1)

    return df_transformed

if __name__ == "__main__":
    config = load_config()

    quant_variables = config['quant_variables']

    train_path = "../outputs/train_df.csv"
    test_path = "../outputs/test_df.csv"

    # load train and test dataframes
    train_df = pd.read_csv(train_path)
    train_df = train_df.set_index("projectid")
    test_df = pd.read_csv(test_path)
    test_df = test_df.set_index("projectid")

    # balance and split the data set by poverty level
    for pov_lvl, pov_col_name in config["poverty_columns"].items():
        smote_dataset = smote_balancing(train_df, pov_col_name, config)

        # scale the data
        smote_dataset_scaled = scale_quant_vars(smote_dataset, quant_variables, pov_lvl)
        
        pov_test_dataset = test_df[test_df[pov_col_name] == True]
        pov_test_dataset = pov_test_dataset.drop(columns=[*config["poverty_columns"].values()], axis=1)

        # scale the data
        pov_test_dataset_scaled = scale_quant_vars(pov_test_dataset, quant_variables)

        smote_path = f"../outputs/{pov_lvl}_pov_lvl_train_df.csv"
        pov_test_path = f"../outputs/{pov_lvl}_pov_lvl_test_df.csv"
        smote_dataset_scaled.to_csv(smote_path, index=False)
        pov_test_dataset_scaled.to_csv(pov_test_path, index=True)

    # scale the data
    train_df_scaled = scale_quant_vars(train_df, quant_variables)
    test_df_scaled = scale_quant_vars(test_df, quant_variables)

    train_df_scaled.to_csv(train_path, index=False)
    test_df_scaled.to_csv(test_path, index=True)

    print(f"Data balancing and segmenting complete.")