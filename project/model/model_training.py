import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import json
import joblib

def train_model(model_type, X_train, y_train, X_test):
    '''
    model_type: string name
    X_train: x training data (pandas dataframe)
    y_train: y training data (pandas dataframe)
    X_test: x testing data (pandas dataframe)
    '''
    if model_type == "random_forest":
        # random forest
        classifier = RandomForestClassifier()

    if model_type == "logistic_regression":
        classifier = LogisticRegression()

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

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

    y_pred = {}
    for model_type in models:
        y_pred[model_type] = train_model(model_type, X_train, y_train, X_test)
    
        pred = pd.DataFrame.from_dict(y_pred)
        
        pred_path = "../outputs/pred.csv"
        pred.to_csv(pred_path, index=False)

    print(f"Model training complete.")
