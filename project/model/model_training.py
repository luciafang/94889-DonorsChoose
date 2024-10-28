import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

    X_train = pd.read_csv(x_train_path)
    X_train = X_train.set_index('projectid')

    y_train = pd.read_csv(y_train_path).squeeze().array
    
    X_test = pd.read_csv(x_test_path)
    X_test = X_test.set_index("projectid")

    config = load_config()
    
    models = config["models"]

    split_by_poverty = config["split_by_poverty"]

    # split by poverty type, refer to config
    if split_by_poverty == "true":
        for pov_lvl, pov_col_name in config["poverty_columns"].items():
            y_pred = {}
            for model_type in models:
                X_res, y_res, X_test_filt = smote_balancing(X_train, y_train, X_test, pov_col_name, config)
                x_res_path = f"../outputs/x_res_{pov_lvl}_poverty.csv"
                y_res_path = f"../outputs/y_res_{pov_lvl}_poverty.csv"
                X_res.to_csv(x_res_path, index=False)
                pd.DataFrame(y_res).to_csv(y_res_path, index=False)
                y_pred[model_type] = train_model(model_type, X_res, y_res, X_test_filt, pov_lvl)
                pred = pd.DataFrame.from_dict(y_pred)

                pred_path = f"../outputs/pred_{pov_lvl}_poverty.csv"
                pred.to_csv(pred_path, index=False)

    # create one model of each type that includes all poverty levels as reference
    y_pred = {}
    pov_lvl = "none"
    for model_type in models:
        y_pred[model_type] = train_model(model_type, X_train, y_train, X_test, pov_lvl)
    
        pred = pd.DataFrame.from_dict(y_pred)
        
        pred_path = "../outputs/pred.csv"
        pred.to_csv(pred_path, index=False)

    print(f"Model training complete.")
