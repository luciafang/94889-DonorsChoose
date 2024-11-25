import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json
import joblib

def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def evaluate_fold(y_test, y_pred):
    """
    evaluate the fold in the TimeSeriesSplit

    y_test: the actual y values
    y_pred: the predicted y values
    """
    # accuracy, precision, recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    # roc for all curves?
    return accuracy, precision, recall

def cross_validate(df, model_type, classifier, pov_lvl, output_label):
    """
    runs cross validation with sklearn's TimeSeriesSplit

    df: the dataframe containing the features and the outcome label
    model_type: the name of the model type (string)
    classifier: the actual classifier object
    pov_lvl: the poverty level of the dataframe (string)
    output_label: the name of the output label (string)
    """
    print("Training: ", model_type, "for ", pov_lvl)
    X = df.copy()
    y = df[output_label]
    if output_label in df.columns:
        X = X.drop([output_label] , axis=1)
    if "date_posted" in df.columns:
        X = X.drop(["date_posted"] , axis=1)
    tscv = TimeSeriesSplit(n_splits=5)
    acc = []
    prec = []
    rec = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        accuracy, precision, recall = evaluate_fold(y_test, y_pred)
        acc.append(accuracy)
        prec.append(precision)
        rec.append(recall)

    metrics = {"accuracy" : acc, "precision" : prec, "recall" : rec}
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_path = "../outputs/" + model_type + f"_{pov_lvl}_cv_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    joblib.dump(classifier, "../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")

if __name__ == "__main__":
    train_path = "../outputs/train_df.csv"
    train_df = pd.read_csv(train_path)

    config = load_config()
    models = config["models"]
    split_by_poverty = config["split_by_poverty"]

    for model_type in models:
        pov_lvl = "none"
        if model_type == "baseline":
            continue
        elif model_type == "random_forest":
            classifier = RandomForestClassifier()
            cross_validate(train_df, model_type, classifier, pov_lvl, "fully_funded")
        elif model_type == "logistic_regression":
            classifier = LogisticRegression(max_iter=1000, solver="liblinear")
            cross_validate(train_df, model_type, classifier, pov_lvl, "fully_funded")
        elif model_type == "svm":
            classifier = svm.LinearSVC()
            cross_validate(train_df, model_type, classifier, pov_lvl, "fully_funded")
            calib_classifier = CalibratedClassifierCV(classifier, method='sigmoid')
            cross_validate(train_df, model_type + "_calibrated", calib_classifier, pov_lvl, "fully_funded")
        # split by poverty type, refer to config
        if split_by_poverty == "true":
            for pov_lvl, pov_col_name in config["poverty_columns"].items():
                pov_train_df_path = f"../outputs/{pov_lvl}_pov_lvl_train_df.csv"
                pov_train_df = pd.read_csv(pov_train_df_path)
                cross_validate(pov_train_df, model_type, classifier, pov_lvl, "fully_funded")
                if "svm" in model_type:
                    cross_validate(pov_train_df, model_type + "_calibrated", calib_classifier, pov_lvl, " ")

    print(f"Model training complete.")