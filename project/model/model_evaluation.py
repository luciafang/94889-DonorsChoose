# metrics: accuracy, precision, recall?
from sklearn.metrics import accuracy_score, precision_score, recall_score, PrecisionRecallDisplay, roc_curve, auc, RocCurveDisplay
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    y_test_path = "../outputs/y_test.csv"
    y_pred_path = "../outputs/pred.csv"

    y_test = pd.read_csv(y_test_path)
    y_pred = pd.read_csv(y_pred_path)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    precision = precision_score(y_test, y_pred)
    print("Precision:", precision)

    recall = recall_score(y_test, y_pred)
    print("Recall:", recall)

    pr_display = PrecisionRecallDisplay.from_predictions(
        y_test, y_pred, name="Random Forest"
    )

    pr_display.ax_.set_title("2-class Precision-Recall curve")

    pr_display.figure_.savefig('../figures/rf_precision_recall.jpg')

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                               estimator_name='Random Forest')
    roc_display.plot().figure_.savefig('../figures/rf_roc.jpg')

    print(f"Model evaluation complete.")