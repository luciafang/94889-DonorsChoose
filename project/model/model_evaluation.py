# metrics: accuracy, precision, recall?
from sklearn.metrics import accuracy_score, precision_score, recall_score, PrecisionRecallDisplay, roc_curve, auc, RocCurveDisplay
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np

def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def metrics(y_test, y_pred, model_type):
    num_splits = y_test.shape[1]
    metrics = {}
    all_acc = []
    all_prec = []
    all_rec = []
    all_fpr = []
    all_tpr = []
    all_auc = []

    for i in range(num_splits):
        y_test_i = y_test[str(i)]
        y_pred_i = y_pred[str(i)]

        accuracy_i = accuracy_score(y_test_i, y_pred_i)
        precision_i = precision_score(y_test_i, y_pred_i)
        recall_i = recall_score(y_test_i, y_pred_i)

        all_acc.append(accuracy_i)
        all_prec.append(precision_i)
        all_rec.append(recall_i)

        fpr, tpr, thresholds = roc_curve(y_test_i, y_pred_i)
        roc_auc = auc(fpr, tpr)

        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_auc.append(roc_auc)

        # Plot ROC curve for this fold
        plt.plot(fpr, tpr)

    plt.savefig("../figures/" + model_type + "_roc.jpg")

    metrics["accuracy"] = all_acc
    metrics["precision"] = all_prec
    metrics["recall"] = all_rec

    print("Avg Accuracy:", np.mean(all_acc))
    print("Avg Precision:", np.mean(all_prec))
    print("Avg Recall:", np.mean(all_rec))


if __name__ == "__main__":
    config = load_config()
    models = config["models"]

    for model_type in models:
        y_test_path = "../outputs/" + model_type + "_true_vals.csv"
        y_pred_path = "../outputs/" + model_type + "_pred.csv"

        y_test = pd.read_csv(y_test_path)
        y_pred = pd.read_csv(y_pred_path)

        print(model_type)
        metrics(y_test, y_pred, model_type)

    print(f"Model evaluation complete.")