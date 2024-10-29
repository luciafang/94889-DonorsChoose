import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
import json
import joblib
from imblearn.over_sampling import SMOTE


def smote_balancing(X_train, y_train, X_test, pov_col_name, config):
    '''
    Use SMOTE to resample/upsample data to balance the two classes
    model_type: string name
    X_train: x training data (pandas dataframe)
    y_train: y training data (pandas dataframe)
    X_test: x testing data (pandas dataframe)
    '''
    smote = SMOTE(random_state=42)

    X = X_train[X_train[pov_col_name] == True].drop(columns=[
        *config["poverty_columns"].values()], axis=1)

    y = y_train[X_train[pov_col_name] == True]

    X_test = X_test[X_test[pov_col_name] == True].drop(columns=[
        *config["poverty_columns"].values()], axis=1)

    X_res, y_res = smote.fit_resample(X, y)
    print(f'SMOTE upsampled from {len(y)} to {len(y_res)} samples')
    return X_res, y_res, X_test


def train_model(model_type, X_train, y_train, X_test, pov_lvl):
    '''
    model_type: string name
    X_train: x training data (pandas dataframe)
    y_train: y training data (pandas dataframe)
    X_test: x testing data (pandas dataframe)
    '''
    if model_type == "random_forest":
        # random forest
        classifier = RandomForestClassifier(random_state=42)

    if model_type == "logistic_regression":
        classifier = LogisticRegression(random_state=42)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    if pov_lvl != "none":
        joblib.dump(classifier, "../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")
    else:
        joblib.dump(classifier, "../outputs/" + model_type + ".pkl")

    return y_pred


def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    x_train_path = "../outputs/x_train.csv"
    y_train_path = "../outputs/y_train.csv"
    x_test_path = "../outputs/x_test.csv"

    train_test_path = "../outputs/train_test_df.csv"
    train_test_df = pd.read_csv(train_test_path)
    train_test_df = train_test_df.set_index('projectid')

    config = load_config()
    models = config["models"]
    split_by_poverty = config["split_by_poverty"]

    # already sorted by date
    X = train_test_df.drop('fully_funded', axis=1)
    y = train_test_df['fully_funded']

    tscv = TimeSeriesSplit(n_splits=5)

    pov_lvl = "none"
    # Perform cross-validation using time series split
    # create one model of each type that includes all poverty levels as reference
    for model_type in models:
        i = 0
        model_preds = {}
        model_true_values = {}
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            y_pred = train_model(model_type, X_train, y_train, X_test, pov_lvl)
            model_preds[str(i)] = y_pred
            model_true_values[str(i)] = y_test.values
            i += 1
        model_preds_df = pd.DataFrame.from_dict(model_preds)
        model_true_values_df = pd.DataFrame.from_dict(model_true_values)

        pred_path = "../outputs/" + model_type + "_pred.csv"
        true_path = "../outputs/" + model_type + "_true_vals.csv"

        model_preds_df.to_csv(pred_path, index=False)
        model_true_values_df.to_csv(true_path, index=False)

    # split by poverty type, refer to config
    if split_by_poverty == "true":
        for pov_lvl, pov_col_name in config["poverty_columns"].items():
            for model_type in models:
                i = 0
                model_preds = {}
                model_true_values = {}
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    # this does not work because smote balancing messes with the number of train/test and there should be the same number in each split
                    # maybe smote balancing should be done earlier?
                    X_res, y_res, X_test_filt = smote_balancing(X_train, y_train, X_test, pov_col_name, config)

                    y_pred = train_model(model_type, X_res, y_res, X_test_filt, pov_lvl)
                    model_preds[str(i)] = y_pred
                    model_true_values[str(i)] = y_test
                    i += 1
                model_preds_df = pd.DataFrame.from_dict(model_preds)
                model_true_values_df = pd.DataFrame.from_dict(model_true_values)

                pred_path = "../outputs/" + model_type + "_" + pov_lvl + "_pred.csv"
                true_path = "../outputs/" + model_type + "_" + pov_lvl + "_true_vals.csv"

                model_preds_df.to_csv(pred_path, index=False)
                model_true_values_df.to_csv(true_path, index=False)

    print(f"Model training complete.")