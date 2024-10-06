import pandas as pd
import json
import numpy as np
import os

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

    # Fill missing values for 'teacher_referred_count' with 0
    if 'teacher_referred_count' in df.columns:
        missing_teacher_referred = df['teacher_referred_count'].isnull().sum()
        df['teacher_referred_count'] = df['teacher_referred_count'].fillna(0)
        print(f"Filled {missing_teacher_referred} missing values in 'teacher_referred_count' with 0.")
    
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
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df_encoded

# Load configuration from JSON
def load_config(config_file="../config.json"):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    config = load_config()
    
    selected_dataset_path = "../outputs/selected_dataset.csv"
    df = pd.read_csv(selected_dataset_path)
    initial_row_count = df.shape[0]
    print(f"Initial number of rows: {initial_row_count}")

    # Clean the dataset
    df_cleaned = clean_and_fill_data(df)
    df_cleaned = remove_outliers(df_cleaned)

    # One-hot encode specified categorical features
    df_cleaned = one_hot_encode(df_cleaned, config['one_hot_encode_features'])

    # Print final row count
    final_row_count = df_cleaned.shape[0]
    print(f"Rows count after cleaning: {final_row_count} ({final_row_count-initial_row_count}).")
    
    # Save the cleaned dataset
    cleaned_dataset_path = "../outputs/cleaned_dataset.csv"
    output_dir = os.path.dirname(cleaned_dataset_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_cleaned.to_csv(cleaned_dataset_path, index=False)
    print(f"Dataset cleaned and saved as '{cleaned_dataset_path}'.")
