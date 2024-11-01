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

def metrics(y_test, y_pred, y_pred_probs, model_type):
    # pr at k
    pre_rec(y_test, y_pred_probs, model_type, k_inc=0.01)
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
        y_pred_prob_i = y_pred_probs[str(i)]

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
        plt.plot(fpr, tpr, label=f'Fold {i} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_type} ROC Curve with Time Series Cross-Validation")
    plt.legend(loc='lower right')
    plt.savefig("../figures/" + model_type + "_roc.jpg")
    plt.clf() 

    metrics["accuracy"] = all_acc
    metrics["precision"] = all_prec
    metrics["recall"] = all_rec

    print("Avg Accuracy:", np.mean(all_acc))
    print("Avg Precision:", np.mean(all_prec))
    print("Avg Recall:", np.mean(all_rec))

#Find precision and recall (true values, scores, k_increase)
def pre_rec(y_test, y_pred_prob, model_type, k_inc=0.01):
    num_splits = y_test.shape[1]
    # somehow attach test to probabilities for each fold
    sorted_true_values = {}
    for i in range(num_splits):
        y_test_i = y_test[str(i)]
        y_pred_i = y_pred_prob[str(i)]

        fold_i = {"true_vals": y_test_i, "probs": y_pred_i}
        fold_i_df = pd.DataFrame.from_dict(fold_i)
        fold_i_df = fold_i_df.sort_values(by='probs', ascending=False)
        sorted_true_values[i] = fold_i_df["true_vals"]

    sorted_true_values = pd.DataFrame.from_dict(sorted_true_values)
    # print(sorted_true_values)
    # for each fold, calculate avg precision and recalls

    # Define percentages of top k predictions (from top 1% to top 100%, increasing by k%)
    percentages_k = np.arange(k_inc, (1.0+k_inc), k_inc)

    # Lists to store precision and recall for each k%
    precision_k = []
    recall_k = []
    
    # Calculate precision and recall for each k%
    for perc in percentages_k:
        top_k = int(perc * len(sorted_true_values))
        y_true_k = sorted_true_values.iloc[:top_k]

        # Calculate avg precision and recall for top k%
        actual_positives = np.sum(sorted_true_values, axis=0)
        predicted_positives = top_k
        pre_k = []
        rec_k = []
        for i in range(num_splits):
            true_positives = np.sum(y_true_k.iloc[:, i], axis=0)

            
            pre_k_i = float(true_positives / predicted_positives) if predicted_positives > 0 else 0
            rec_k_i = float(true_positives / actual_positives[i]) if actual_positives[i] > 0 else 0

            pre_k.append(pre_k_i)
            rec_k.append(rec_k_i)
        
        precision_k.append(np.average(pre_k))
        recall_k.append(np.average(rec_k))

    #Plot 
    fig, ax1 = plt.subplots()

    # Plotting the Recall on the primary y-axis
    ax1.plot(percentages_k*100, recall_k, 'g-', label='Recall')  # 'g-' means green solid line
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Recall', color='g')
    ax1.tick_params(axis='y', labelcolor='g')  # Change the color of the ticks to match the line
    ax1.legend(loc='upper right')  # Add legend for the primary axis

    # Create a second y-axis and plot the precision
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    ax2.plot(percentages_k*100, precision_k, 'b-', label='Precision')  # 'b-' means blue solid line
    ax2.set_ylabel('Precision', color='b')
    ax2.tick_params(axis='y', labelcolor='b')  # Change the color of the ticks to match the line
    ax2.legend(loc='lower right')  # Add legend for the secondary axis

    # Save the plot
    plt.title('PR-k curve')
    plt.savefig("../figures/" + model_type + "_pr_k_plot.jpg")


    # return (precision_k, recall_k)

if __name__ == "__main__":
    config = load_config()
    models = config["models"]

    split_by_poverty = config["split_by_poverty"]

    for model_type in models:
        pov_lvl = "none"
        y_pred_path = "../outputs/" + model_type + f"_{pov_lvl}_pred.csv"
        y_pred_prob_path = "../outputs/" + model_type + f"_{pov_lvl}_pred_probs.csv"
        y_test_path = "../outputs/" + model_type + f"_{pov_lvl}_true_vals.csv"

        y_test = pd.read_csv(y_test_path)
        y_pred = pd.read_csv(y_pred_path)
        y_pred_probs = pd.read_csv(y_pred_prob_path)

        print(model_type)
        metrics(y_test, y_pred, y_pred_probs, model_type)
        if split_by_poverty == "true":
            for pov_lvl, pov_col_name in config["poverty_columns"].items():
                print(pov_col_name)
                y_pred_path = "../outputs/" + model_type + f"_{pov_lvl}_pred.csv"
                y_pred_prob_path = "../outputs/" + model_type + f"_{pov_lvl}_pred_probs.csv"
                y_test_path = "../outputs/" + model_type + f"_{pov_lvl}_true_vals.csv"

                y_test = pd.read_csv(y_test_path)
                y_pred = pd.read_csv(y_pred_path)
                y_pred_probs = pd.read_csv(y_pred_prob_path)
                metrics(y_test, y_pred, y_pred_probs, model_type)

        print(f"Model evaluation complete.")