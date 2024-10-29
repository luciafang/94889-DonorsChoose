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

    # classifier = joblib.load("../outputs/" + model_type + ".pkl")
    # pr_at_k(classifier, X_test, y_test, model_type)

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

    for model_type in models:
        y_test_path = "../outputs/" + model_type + "_true_vals.csv"
        y_pred_path = "../outputs/" + model_type + "_pred.csv"

        y_test = pd.read_csv(y_test_path)
        y_pred = pd.read_csv(y_pred_path)

        print(model_type)
        metrics(y_test, y_pred, model_type)

        print(f"Model evaluation complete.")