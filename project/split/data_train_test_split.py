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
    # for training, we focus on projects date_posted before 2014-01-12
    # rest is testing (2014-01-12->2014-05-11, which is max date)
    df['date_posted'] = pd.to_datetime(df['date_posted'], format='%Y-%m-%d', errors='coerce')
    max_date = df['date_posted'].max()
    cutoff_date = max_date - pd.DateOffset(months=4)
    train_df = df[df['date_posted'] <= cutoff_date].copy().drop('date_posted', axis=1)
    test_df = df[df['date_posted'] > cutoff_date].copy().drop('date_posted', axis=1)

    # also there may be projects that didn't finish their funding rounds (-1) dropping the unknowns
    filtered_train_df = train_df[train_df['fully_funded'] != -1]
    y_train = filtered_train_df['fully_funded']
    X_train = filtered_train_df.drop('fully_funded', axis=1)

    # this should all be -1 for y_test, since we don't know the results yet (for recommendation)
    y_test = test_df['fully_funded']
    X_test = test_df.drop('fully_funded', axis=1)

    print('Shape of training set:', X_train.shape)
    print('Shape of test set:', X_test.shape)

    # X = df.drop('fully_funded', axis=1)
    # y = df['fully_funded']
    #
    # # Split the data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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