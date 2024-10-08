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

def save_feature_importance_plot(classifier, X_train):
    importances = classifier.feature_importances_

    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))

    _ = ax.set_yticklabels(np.array(X_train.columns)[indices])

    plt.tight_layout()
    
    fig.savefig('../figures/feature_importances.jpg')

if __name__ == "__main__":
    x_train_path = "../outputs/x_train.csv"
    X_train = pd.read_csv(x_train_path)
    X_train = X_train.set_index('projectid')

    config = load_config()
    models = config["models"]
    for model_type in models:
        classifier = joblib.load("../outputs/" + model_type + ".pkl")
        save_feature_importance_plot(classifier, X_train)
    