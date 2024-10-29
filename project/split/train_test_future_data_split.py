import sklearn as sk
import pandas as pd
import os

# train test split, cross validation with time series (date_posted) ?
# just a set up to get started, will update once OHE done
from sklearn.model_selection import train_test_split

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

if __name__ == "__main__":
    cleaned_dataset_path = "../outputs/cleaned_dataset.csv"
    df = pd.read_csv(cleaned_dataset_path)

    train_test_df, future_df = split_data(df)

    train_test_path = "../outputs/train_test_df.csv"
    future_path = "../outputs/future_df.csv"
    
    train_test_df.to_csv(train_test_path, index=False)
    future_df.to_csv(future_path, index=False)

    print(f"Train/test and future data split complete.")