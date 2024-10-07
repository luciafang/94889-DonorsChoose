import sklearn as sk
import pandas as pd
import os

# train test split, cross validation with time series (date_posted) ?
# just a set up to get started, will update once OHE done
from sklearn.model_selection import train_test_split

def split_data(df, outcome_name):
    '''
    df: pandas dataframe dataset of features and outcome variable
    outcome_name: string name of outcome variable
    '''
    # Split the data into features (X) and target (y)
    X = df.drop('fully_funded', axis=1)
    y = df['fully_funded']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    cleaned_dataset_path = "../outputs/cleaned_dataset.csv"
    df = pd.read_csv(cleaned_dataset_path)

    outcome_name = "fully_funded"

    X_train, X_test, y_train, y_test = split_data(df, outcome_name)


    # Save the training and test datasets
    x_train_path = "../outputs/x_train.csv"
    x_test_path = "../outputs/x_test.csv"
    y_train_path = "../outputs/y_train.csv"
    y_test_path = "../outputs/y_test.csv"
    
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    print(f"Train/test split complete.")