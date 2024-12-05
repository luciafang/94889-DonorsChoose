import pandas as pd
import json
import numpy as np
import os

def get_school_metro(project, df_projects_metro_filled):
  '''
  project: row in projects with metro df
  '''
  new_proj = df_projects_metro_filled[df_projects_metro_filled["projectid"] == project["projectid"]]
  return new_proj["school_metro_filled"].values[0]

def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def clean_and_fill_data(df):
    """
    Remove invalid entries from the DataFrame by checking for missing values,
    dropping duplicates, and dropping rows with any null values.

    :param df: DataFrame to clean
    :param df_name: Name of the DataFrame for display purposes
    :return: Cleaned DataFrame
    """
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values in each column:\n{missing_values[missing_values > 0]}")
    
    # Drop duplicates
    duplicates_count = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"Dropped {duplicates_count} duplicate rows.")

    # Fill missing values with imputation
    config = load_config()
    
    projects_imputation = config["projects_imputation"]
    for imp in projects_imputation:
        nulls = df[imp].isnull().sum()
        if projects_imputation[imp] == "0":
            
            df[imp] = df[imp].fillna(0)
            print(f"Filled {nulls} missing values in {imp} with 0.")
        elif projects_imputation[imp] == "mean":
            df[imp] = df[imp].fillna(df[imp].mean())
            print(f"Filled {nulls} missing values in {imp} with mean.")
        # removed school metro as a feature because we don't have all states/cities data
        # no features are being imputed with "gpt"
        elif projects_imputation[imp] == "gpt":
            selected_dataset_path = "../data/projects_with_metro_gpt.csv"
            df_metro = pd.read_csv(selected_dataset_path)
            df["school_metro"] = df.apply(lambda project: project["school_metro"] if isinstance(project["school_metro"], str) else get_school_metro(project, df_metro), axis=1)
            
            print(f"Filled {nulls} missing values in {imp} with estimate from ChatGPT.")
        # negative 1 fill for unknown not_fully_funded rows (anything that is 2014)
        elif projects_imputation[imp] == "neg1":
            df[imp] = df[imp].fillna(-1)
            print(f"Filled {nulls} missing values in {imp} with -1.")

    # Drop rows with any null values
    null_rows_count = df.isnull().sum().sum()
    df.dropna(inplace=True)
    dropped_null_count = null_rows_count - df.isnull().sum().sum()

    print(f"Dropped {dropped_null_count} rows with null values.")

    return df

def remove_outliers(df):
    '''
    df: dataframe
    cols_to_exclude: list of columns to exclude from outlier removal
    '''
    skipped_columns = ['teacher_referred_count']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    initial_row_count = df.shape[0]

    for col in numeric_cols:
        if col in skipped_columns:
            print(f"Skipping outlier removal for column '{col}'.")
            continue
    
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
    
        # Identify outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = outliers.shape[0]
    
        # Print information about outliers
        if outlier_count > 0:
            print(f"Column '{col}': Found {outlier_count} outliers (lower bound: {lower_bound}, upper bound: {upper_bound}).")
    
        # Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    final_row_count = df.shape[0]
    dropped_outlier_count = initial_row_count - final_row_count
    print(f"Total outliers removed: {dropped_outlier_count}.")

    return df

def one_hot_encode(df, categorical_features):
    """
    One-hot encode the specified categorical features.

    :param df: DataFrame to encode
    :param categorical_features: List of categorical features to encode
    :return: DataFrame with one-hot encoded features
    """
    print(f"One-hot encoding: {categorical_features}")
    df_encoded = pd.get_dummies(df, columns=categorical_features)
    return df_encoded

def combine_poverty_levels(df, replacements):
    """
    replaces poverty level values in poverty level column with sepcified replacement
    df: pandas dataframe 
    replacements: dictionary with the keys being the poverty levels being replaced and the values
    being the poverty levels the key names are being replaced with
    """
    for pov_lvl in replacements:
        df['poverty_level'] = df['poverty_level'].replace(pov_lvl, replacements[pov_lvl])
    return df

if __name__ == "__main__":
    config = load_config()

    categorical_variables = config['one_hot_encode_features']
    
    selected_dataset_path = "../outputs/selected_dataset_with_new_features.csv"
    df = pd.read_csv(selected_dataset_path)
    initial_row_count = df.shape[0]
    print(f"Initial number of rows: {initial_row_count}")

    # Clean the dataset
    df_cleaned = clean_and_fill_data(df)

    # combine low and moderate poverty to low and high and highest poverty to high
    df_cleaned = combine_poverty_levels(df, config["poverty_level_replacements"])

    # One-hot encode specified categorical features
    df_cleaned = one_hot_encode(df_cleaned, categorical_variables)

    # Print final row count
    final_row_count = df_cleaned.shape[0]
    print(f"Rows count after cleaning: {final_row_count} ({final_row_count-initial_row_count}).")
    
    # Save the cleaned dataset
    cleaned_dataset_path = "../outputs/cleaned_dataset.csv"
    output_dir = os.path.dirname(cleaned_dataset_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_cleaned.to_csv(cleaned_dataset_path, index=True)
    print(f"Dataset cleaned and saved as '{cleaned_dataset_path}'.")
