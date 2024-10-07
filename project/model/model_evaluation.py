# metrics: accuracy, precision, recall?
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

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

    print(f"Model evaluation complete.")