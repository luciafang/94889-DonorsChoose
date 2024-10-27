# metrics: accuracy, precision, recall?
from sklearn.metrics import accuracy_score, precision_score, recall_score, PrecisionRecallDisplay, roc_curve, auc, RocCurveDisplay
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
import numpy as np

def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def metrics(models, y_test, y_pred, X_test):
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

        classifier = joblib.load("../outputs/" + model_type + ".pkl")
        pr_at_k(classifier, X_test, y_test, model_type)

def pr_at_k(classifier, X_test, y_test, model_type):
    X_test = X_test.set_index("projectid")
    probabilities = classifier.predict_proba(X_test)

    X_test["pred_prob"] = probabilities[:, 1]
    X_test["true_values"] = y_test.values

    probs_df = X_test[["pred_prob", "true_values"]]
    probs_df = probs_df.sort_values(by='pred_prob', ascending=False)

    precisions = []
    recalls = []
    true_values = probs_df["true_values"].values

    total_true = sum(true_values)
    for i in range(1, len(true_values)+1):
        # precision = sum of trues/count so far
        precision = sum(true_values[:i])/i
        precisions.append(precision)
        # recall = sum of trues/total number of actual trues
        recall = sum(true_values[:i])/total_true
        recalls.append(recall)
    
    # make plot with two y axes
    # Create a figure and axis
    fig, ax1 = plt.subplots()

    k = np.arange(len(true_values))

    # Plot the precision
    ax1.plot(k, precisions, 'g-', label='Precision')  # Green line for precision
    ax1.set_xlabel('k')
    ax1.set_ylabel('Precision', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    # Create a second y-axis
    ax2 = ax1.twinx()
    ax2.plot(k, recalls, 'b-', label='Recall')  # Blue line for recall
    ax2.set_ylabel('Recall', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Add a title and a legend
    plt.title('Precision and Recall vs. Threshold')
    fig.tight_layout()  # To make sure the layout is neat
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot
    plt.savefig("../figures/" + model_type + "_pr_k_plot.jpg")


if __name__ == "__main__":
    config = load_config()
    models = config["models"]

    
    y_test_path = "../outputs/y_test.csv"
    y_pred_path = "../outputs/pred.csv"
    x_test_path = "../outputs/x_test.csv"

    y_test = pd.read_csv(y_test_path)
    y_pred = pd.read_csv(y_pred_path)

    X_test = pd.read_csv(x_test_path)

    metrics(models, y_test, y_pred, X_test)

    print(f"Model evaluation complete.")