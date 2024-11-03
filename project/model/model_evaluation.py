# metrics: accuracy, precision, recall?
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
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

def plot_roc_curve(y_test, y_pred, model_type):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_type} ROC Curve with Time Series Cross-Validation")
    plt.legend(loc='lower right')
    plt.savefig("../figures/" + model_type + "_roc.jpg")
    plt.clf()

def metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

#Find precision and recall (true values, scores, k_increase)
def pre_rec(y_t, y_s, model_type, k_inc=0.01):
    # Sort scores and true labels based on the predicted scores (descending order)
    sorted_i = np.argsort(y_s)[::-1]
    y_true_s = np.array(y_t)[sorted_i]

    # Define percentages of top k predictions (from top 1% to top 100%, increasing by k%)
    percentages_k = np.arange(k_inc, (1.0+k_inc), k_inc)

    # Lists to store precision and recall for each k%
    precision_k = []
    recall_k = []

    # Calculate precision and recall for each k%
    for perc in percentages_k:
        top_k = int(perc * len(y_true_s))
        y_true_k = y_true_s[:top_k]
        
        # Calculate precision and recall for top k%
        true_positives = np.sum(y_true_k)
        predicted_positives = top_k
        actual_positives = np.sum(y_true_s)
        
        pre_k = float(true_positives / predicted_positives) if predicted_positives > 0 else 0
        rec_k = float(true_positives / actual_positives) if actual_positives > 0 else 0
        
        precision_k.append(pre_k)
        recall_k.append(rec_k)

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

    # Show the plot
    plt.title('PR-k curve')
    plt.savefig("../figures/" + model_type + "_prk.jpg")
    plt.clf()

    return (precision_k,recall_k)

def evaluate(test_df, classifier, model_type):
    print(model_type)
    X = test_df.drop(["fully_funded", "date_posted"], axis=1)
    y = test_df["fully_funded"]

    y_pred = classifier.predict(X)
    y_pred_probs = classifier.predict_proba(X)[:, 1]

    pre_rec(y, y_pred_probs, model_type, k_inc=0.01)
    plot_roc_curve(y, y_pred, model_type)
    metrics(y, y_pred)

    return y_pred, y_pred_probs

if __name__ == "__main__":
    config = load_config()
    models = config["models"]

    split_by_poverty = config["split_by_poverty"]

    for model_type in models:
        pov_lvl = "none"
        classifier = joblib.load("../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")

        test_path = "../outputs/test_df.csv"
        test_df = pd.read_csv(test_path)
        test_df = test_df.set_index('projectid')

        evaluate(test_df, classifier, model_type)
        
        # y_pred_path = "../outputs/" + model_type + f"_{pov_lvl}_pred.csv"
        # y_pred_prob_path = "../outputs/" + model_type + f"_{pov_lvl}_pred_probs.csv"
        # y_test_path = "../outputs/" + model_type + f"_{pov_lvl}_true_vals.csv"

        # y_test = pd.read_csv(y_test_path)
        # y_pred = pd.read_csv(y_pred_path)
        # y_pred_probs = pd.read_csv(y_pred_prob_path)

        # print(model_type)
        # metrics(y_test, y_pred, y_pred_probs, model_type)
        # if split_by_poverty == "true":
        #     for pov_lvl, pov_col_name in config["poverty_columns"].items():
        #         print(pov_col_name)
        #         y_pred_path = "../outputs/" + model_type + f"_{pov_lvl}_pred.csv"
        #         y_pred_prob_path = "../outputs/" + model_type + f"_{pov_lvl}_pred_probs.csv"
        #         y_test_path = "../outputs/" + model_type + f"_{pov_lvl}_true_vals.csv"

        #         y_test = pd.read_csv(y_test_path)
        #         y_pred = pd.read_csv(y_pred_path)
        #         y_pred_probs = pd.read_csv(y_pred_prob_path)
        #         metrics(y_test, y_pred, y_pred_probs, model_type)

    print(f"Model evaluation complete.")