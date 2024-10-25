# metrics: accuracy, precision, recall?
from sklearn.metrics import accuracy_score, precision_score, recall_score, PrecisionRecallDisplay, roc_curve, auc, RocCurveDisplay
import pandas as pd
import matplotlib.pyplot as plt
import json

def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def metrics(models, y_test, y_pred):
    for model_type in models:
        y_pred_model = y_pred[model_type]
        print(model_type)

        accuracy = accuracy_score(y_test, y_pred_model)
        print("Accuracy:", accuracy)

        precision = precision_score(y_test, y_pred_model)
        print("Precision:", precision)

        recall = recall_score(y_test, y_pred_model)
        print("Recall:", recall)

        pr_display = PrecisionRecallDisplay.from_predictions(
            y_test, y_pred_model, name="model_type"
        )

        pr_display.ax_.set_title(model_type + "2-class Precision-Recall curve")

        pr_display.figure_.savefig("../figures/" + model_type + "_precision_recall.jpg")

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_model)
        roc_auc = auc(fpr, tpr)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                estimator_name=model_type)
        roc_display.plot().figure_.savefig("../figures/" + model_type + "_roc.jpg")


if __name__ == "__main__":
    config = load_config()
    models = config["models"]

    
    y_test_path = "../outputs/y_test.csv"
    y_pred_path = "../outputs/pred.csv"

    y_test = pd.read_csv(y_test_path)
    y_pred = pd.read_csv(y_pred_path)

    metrics(models, y_test, y_pred)

    print(f"Model evaluation complete.")