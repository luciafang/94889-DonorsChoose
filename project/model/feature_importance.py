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

def save_feature_importance_plot(classifier, X, model_type, pov_lvl):
    importances = classifier.feature_importances_

    indices = np.argsort(importances)[-10:]  # Select top 10 features
    top_features = np.array(X.columns)[indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top_importances)), top_importances, color='skyblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 10 Feature Importances')

    plt.tight_layout()

    # Save the plot
    filename = f"../figures/{model_type}_{pov_lvl}_poverty_feature_importances.jpg" if pov_lvl != "none" \
        else f"../figures/{model_type}_feature_importances.jpg"
    fig.savefig(filename)
    plt.close(fig)

def save_cofficient_plot(classifier, X_train, model_type, pov_lvl):

    coefficients = classifier.coef_[0]
    indices = np.argsort(np.abs(coefficients))[-10:]  # Select top 10 by absolute value
    top_features = np.array(X_train.columns)[indices]
    top_coefficients = coefficients[indices]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['green' if coef > 0 else 'red' for coef in top_coefficients]  # Positive vs negative
    ax.barh(top_features, top_coefficients, color=colors)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Top 10 Coefficients')

    plt.tight_layout()

    # Save the plot
    filename = f"../figures/{model_type}_{pov_lvl}_poverty_coeff_plot.jpg" if pov_lvl != "none" \
        else f"../figures/{model_type}_coeff_plot.jpg"
    fig.savefig(filename)
    plt.close(fig)
if __name__ == "__main__":
    config = load_config()
    models = config["models"]
    split_by_poverty = config["split_by_poverty"]

    df_path = "../outputs/train_df.csv"
    df = pd.read_csv(df_path)
    X = df.copy()
    if "fully_funded" in df.columns:
        X = X.drop(["fully_funded"] , axis=1)
    if "date_posted" in df.columns:
        X = X.drop(["date_posted"] , axis=1)
    if split_by_poverty == "true":
        for pov_lvl, pov_col_name in config["poverty_columns"].items():
            # X_res already removed projectid column thus removing it
            pov_train_df_path = f"../outputs/{pov_lvl}_pov_lvl_train_df.csv"
            pov_train_df = pd.read_csv(pov_train_df_path)
            if "fully_funded" in pov_train_df.columns:
                pov_train_df = pov_train_df.drop(["fully_funded"] , axis=1)
            if "date_posted" in pov_train_df.columns:
                pov_train_df = pov_train_df.drop(["date_posted"] , axis=1)
            for model_type in models:
                if model_type == "baseline":
                    continue
                classifier = joblib.load("../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")
                if model_type == "random_forest":
                    save_feature_importance_plot(classifier, pov_train_df, model_type, pov_lvl)
                if model_type == "logistic_regression":
                    save_cofficient_plot(classifier, pov_train_df, model_type, pov_lvl)
                if model_type == "svm":
                    save_cofficient_plot(classifier, pov_train_df, model_type, pov_lvl)
    
    # for model with all poverty levels
    models = config["models"]
    pov_lvl = "none"
    for model_type in models:
        if model_type == "baseline":
                    continue
        classifier = joblib.load("../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")
        if model_type == "random_forest":
            save_feature_importance_plot(classifier, X, model_type, pov_lvl)
        if model_type == "logistic_regression":
            save_cofficient_plot(classifier, X, model_type, pov_lvl)
        if model_type == "svm":
            save_cofficient_plot(classifier, X, model_type, pov_lvl)