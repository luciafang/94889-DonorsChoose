import sklearn as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def save_feature_importance_plot(classifier, X_train, model_type, pov_lvl):
    importances = classifier.feature_importances_

    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))

    _ = ax.set_yticklabels(np.array(X_train.columns)[indices])

    plt.tight_layout()
    
    fig.savefig("../figures/" + model_type + f"_{pov_lvl}_poverty_feature_importances.jpg")

def save_cofficient_plot(classifier, X_train, model_type, pov_lvl):
    # Get the coefficients and feature names
    coefficients = classifier.coef_[0]
    feature_names = X_train.columns

    # Create a horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coefficients)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Logistic Regression Coefficients')
    plt.tight_layout()
    plt.savefig("../figures/" + model_type + f"_{pov_lvl}_poverty_coeff_plot.jpg")

if __name__ == "__main__":
    config = load_config()
    models = config["models"]
    for pov_lvl, pov_col_name in config["poverty_columns"].items():
        x_train_path = f"../outputs/x_res_{pov_lvl}_poverty.csv"
        X_train = pd.read_csv(x_train_path)
        # X_res already removed projectid column thus removing it
        # X_train = X_train.set_index('projectid')
        for model_type in models:
            classifier = joblib.load("../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")
            if model_type == "random_forest":
                save_feature_importance_plot(classifier, X_train, model_type, pov_lvl)
            if model_type == "logistic_regression":
                save_cofficient_plot(classifier, X_train, model_type, pov_lvl)