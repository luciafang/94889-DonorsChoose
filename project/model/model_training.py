import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
import json
import joblib

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
    y_pred_probs = classifier.predict_proba(X_test)[:, 1]

    if pov_lvl != "none":
        joblib.dump(classifier, "../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")
    else:
        joblib.dump(classifier, "../outputs/" + model_type + ".pkl")

    return y_pred, y_pred_probs


def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def cross_validate(df, model_type, pov_lvl, output_label):
    X = df.drop(output_label, axis=1)
    y = df[output_label]
    tscv = TimeSeriesSplit(n_splits=5)

    i = 0
    model_preds = {}
    model_pred_probs = {}
    model_true_values = {}
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        y_pred, y_pred_probs = train_model(model_type, X_train, y_train, X_test, pov_lvl)
        model_preds[str(i)] = y_pred
        model_pred_probs[str(i)] = y_pred_probs
        model_true_values[str(i)] = y_test.values
        i += 1
    model_preds_df = pd.DataFrame.from_dict(model_preds)
    model_pred_probs_df = pd.DataFrame.from_dict(model_pred_probs)
    model_true_values_df = pd.DataFrame.from_dict(model_true_values)

    pred_path = "../outputs/" + model_type + f"_{pov_lvl}_pred.csv"
    pred_prob_path = "../outputs/" + model_type + f"_{pov_lvl}_pred_probs.csv"
    true_path = "../outputs/" + model_type + f"_{pov_lvl}_true_vals.csv"

    model_preds_df.to_csv(pred_path, index=False)
    model_pred_probs_df.to_csv(pred_prob_path, index=False)
    model_true_values_df.to_csv(true_path, index=False)

if __name__ == "__main__":
    train_test_path = "../outputs/train_test_df.csv"
    train_test_df = pd.read_csv(train_test_path)
    train_test_df = train_test_df.set_index('projectid')

    config = load_config()
    models = config["models"]
    split_by_poverty = config["split_by_poverty"]

    for model_type in models:
        pov_lvl = "none"
        cross_validate(train_test_df, model_type, pov_lvl, "fully_funded")
        # split by poverty type, refer to config
        if split_by_poverty == "true":
            for pov_lvl, pov_col_name in config["poverty_columns"].items():
                train_df_path = f"../outputs/{pov_lvl}_pov_lvl_train_test_df.csv"
                train_df = pd.read_csv(train_df_path)
                cross_validate(train_test_df, model_type, pov_lvl, "fully_funded")

    print(f"Model training complete.")