# metrics: accuracy, precision, recall?
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
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

def plot_roc_curve(y_test, y_pred, model_type, pov_lvl):
    """
    Plots ROC curve for output

    y_test: actual outcome labels for test set
    y_pred: predicted outcome labels for test set
    model_type: the string name of the model
    pov_lvl: the poverty level for the model
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_type.replace("_", " ").title()} Model {pov_lvl.capitalize()} Poverty Level ROC Curve")
    plt.legend(loc='lower right')
    plt.savefig("../figures/" + model_type  + "_" + pov_lvl +  "_roc.jpg")
    plt.clf()

def metrics(y_test, y_pred):
    """
    Prints accuracy, precision, and recall for model output

    y_test: actual outcome labels for test set
    y_pred: predicted outcome labels for test set
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

#F ind precision and recall (true values, scores, k_increase)
def pre_rec(y_t, y_s, model_type, pov_lvl, k_inc=0.01):
    """
    Plots PR-k curve for output

    y_t: actual outcome labels for test set
    y_s: predicted outcome label scores for test set
    model_type: the string name of the model
    pov_lvl: the poverty level for the model
    k_inc: the increase percentage for k
    """
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
    plt.title(f"{model_type.replace("_", " ").title()} Model {pov_lvl.capitalize()} Poverty Level PR-k Curve")
    plt.savefig("../figures/" + model_type + "_" + pov_lvl + "_prk.jpg")
    plt.clf()

    return (precision_k,recall_k)

def baseline_predict(df, sort_col):
    """
    Predicts outcome for baseline model

    df: dataframe of test set
    sort_col: the column to sort the test set by for predictions
    """
    copy_df = df.copy()

    # use specified column sort_col to sort dataset and predict outcome variables
    sorted_df = copy_df.sort_values(by=sort_col, ascending=False)

    # get values in column
    sort_col_df = sorted_df[sort_col].to_frame()

    # use .3 bc 30% of projects don't get funded, so want to predict top 30 in sort_col as not getting funded
    split_val = round(.3 * len(sort_col_df))

    copy_df["not_fully_funded_pred"] = 1
    
    # predict all the projects with a value less than the split_val as getting funded, should be about 70% of projects
    copy_df.loc[copy_df[sort_col] < sort_col_df.iloc[split_val].values[0], "not_fully_funded_pred"] = 0

    # return y_test, y_pred
    return copy_df["not_fully_funded_pred"]

def print_model_metrics(y_test, y_pred, model_type, pov_lvl):
    """Print model metrics in a formatted table"""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Create a separator line
    separator = "-" * 60
    
    # Print header
    if pov_lvl != "none":
        print(f"\n{separator}")
        print(f"Poverty Level: {pov_lvl.upper()}")
    print(f"{separator}")
    print(f"Model: {model_type.upper()}")
    print(f"{separator}")
    
    # Create metrics table
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall'],
        'Value': [accuracy, precision, recall]
    })
    
    # Format the values to 4 decimal places
    metrics_df['Value'] = metrics_df['Value'].map('{:.4f}'.format)
    
    # Add padding for better alignment
    col_width = max(len(word) for row in metrics_df.values for word in row) + 2
    print(f"{'Metric':<{col_width}}{'Value':<{col_width}}")
    print("-" * (col_width * 2))
    
    for _, row in metrics_df.iterrows():
        print(f"{row['Metric']:<{col_width}}{row['Value']:<{col_width}}")
    
    print(separator)

def print_summary_table(all_results):
    """Print summary table for all models and poverty levels"""
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL MODELS AND POVERTY LEVELS")
    print("=" * 80)
    
    # Create a DataFrame for easy formatting
    summary_data = []
    for pov_lvl, models in all_results.items():
        for model, metrics in models.items():
            summary_data.append({
                'Poverty Level': pov_lvl.upper(),
                'Model': model.upper(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Format numeric columns
    for col in ['Accuracy', 'Precision', 'Recall']:
        summary_df[col] = summary_df[col].map('{:.4f}'.format)
    
    print(summary_df.to_string(index=False))
    print("=" * 80)
    
    # Save to CSV
    summary_df.to_csv("../outputs/model_metrics_summary.csv", index=False)

# Modify the evaluate function to use the new printing function
def evaluate(test_df, classifier, model_type, pov_lvl):
    """
    Evaluate test dataset for specified classifier

    classifier: the classifier object
    model_type: the string name of the model
    pov_lvl: the poverty level for the model
    """
    X = test_df.copy()
    y = test_df["not_fully_funded"]
    if "not_fully_funded" in test_df.columns:
        X = X.drop(["not_fully_funded"] , axis=1)
    if "date_posted" in test_df.columns:
        X = X.drop(["date_posted"] , axis=1)

    if model_type != "baseline":
        y_pred = classifier.predict(X)
        if "svm" in model_type:
            calib_classifier = joblib.load("../outputs/" + model_type + f"_calibrated_{pov_lvl}_poverty.pkl")
            y_pred_probs = calib_classifier.predict_proba(X)[:, 1]
        else:
            y_pred_probs = classifier.predict_proba(X)[:, 1]
        pre_rec(y, y_pred_probs, model_type, pov_lvl, k_inc=0.01)
    else:
        y_pred = baseline_predict(X, "total_price_excluding_optional_support")
        y_pred_probs = []
    
    plot_roc_curve(y, y_pred, model_type, pov_lvl)
    print_model_metrics(y, y_pred, model_type, pov_lvl)
    
    return y, y_pred, y_pred_probs, X

def plot_false_discovery_rate_ref(model_names, model_outputs, test_df, protected_cols, protected_col_name, pov_lvl):
    """
    Plots FDR disparity between the models

    model_names: list of the models
    model_outputs: dictionary of outputs of the models
    test_df: dataframe of test set
    stem_cols: list of primary focus subjects that are STEM
    pov_lvl: the poverty level for the model
    """
    test_df["is_protected"] = test_df[protected_cols].any(axis=1)
    non_protected_cols = [col for col in test_df.columns if protected_col_name in col and col not in protected_cols]
    test_df["not_protected"] = test_df[non_protected_cols].any(axis=1)
    ref_mask = test_df["not_protected"] == True
    protect_mask = test_df["is_protected"] == True
    model_fdrs = []
    model_prec = []
    for output in model_outputs:
        tn, fp, fn, tp = confusion_matrix(model_outputs[output]["y_test"][ref_mask], model_outputs[output]["y_pred"][ref_mask]).ravel()
        fdr_ref = fp / (fp + tp)

        tn, fp, fn, tp = confusion_matrix(model_outputs[output]["y_test"][protect_mask], model_outputs[output]["y_pred"][protect_mask]).ravel()
        fdr_protect = fp / (fp + tp)

        model_fdrs.append(fdr_protect/fdr_ref)

        precision = precision_score(model_outputs[output]["y_test"], model_outputs[output]["y_pred"])
        model_prec.append(precision)

    # Plotting
    plt.figure(figsize=(8, 6))
    colors = ["blue", "orange", "purple", "pink"]
    i = 0
    for fdr in model_fdrs:
        prec = model_prec[i]
        model = model_names[i]
        color = colors[i]
        i += 1

        # Scatter plot with different colors for each model
        plt.scatter(prec, fdr, color=color, label=model, s=100)  # `s` controls the size of points

    # Add labels and legend
    plt.xlabel('Precision')
    plt.ylabel('False Discovery Rate (FDR) Disparity')
    plt.title(f"{protected_col_name.replace("_", " ").title()} FDR Disparity vs Precision for {pov_lvl.capitalize()} Poverty Level")
    plt.legend()
    plt.xlim(0, 2)
    plt.ylim(0, 2) 

    # Save the plot
    plt.savefig("../figures/" + pov_lvl + "_" + protected_col_name + "_fdr_disparity_plot.jpg")
    plt.clf()

def plot_recall_disparity(model_names, model_outputs, test_df, protected_cols, protected_col_name, pov_lvl):
    """
    Plots recall disparity between the models

    model_names: list of the models
    model_outputs: dictionary of outputs of the models
    test_df: dataframe of test set
    stem_cols: list of primary focus subjects that are STEM
    pov_lvl: the poverty level for the model
    """
    test_df["is_protected"] = test_df[protected_cols].any(axis=1)
    non_protected_cols = [col for col in test_df.columns if protected_col_name in col and col not in protected_cols]
    test_df["not_protected"] = test_df[non_protected_cols].any(axis=1)
    ref_mask = test_df["not_protected"] == True
    protect_mask = test_df["is_protected"] == True

    model_recalls = []
    model_prec = []
    for output in model_outputs:
        recall_ref = recall_score(model_outputs[output]["y_test"][ref_mask], model_outputs[output]["y_pred"][ref_mask])
        recall_protect = recall_score(model_outputs[output]["y_test"][protect_mask], model_outputs[output]["y_pred"][protect_mask])
        model_recalls.append(recall_protect/recall_ref)

        precision = precision_score(model_outputs[output]["y_test"], model_outputs[output]["y_pred"])
        model_prec.append(precision)

    
    plt.figure(figsize=(8, 6))
    colors = ["blue", "orange", "purple", "pink"]
    i = 0
    for recall in model_recalls:
        prec = model_prec[i]
        model = model_names[i]
        color = colors[i]
        i += 1
        # Scatter plot with different colors for each model
        plt.scatter(prec, recall, color=color, label=model, s=100)

    # Add labels and legend
    plt.xlabel('Precision')
    plt.ylabel('Recall Disparity')
    plt.title(f"{protected_col_name.replace("_", " ").title()} Recall Disparity vs Precision for {pov_lvl.capitalize()} Poverty Level")
    plt.legend()
    plt.xlim(0, 2)
    plt.ylim(0, 2)

    # Save the plot
    plt.savefig("../figures/" + pov_lvl + "_" + protected_col_name + "_recall_disparity_plot.jpg")
    plt.clf()

def collect_temporal_performance(test_df, classifier, model_type):
    """
    Collect model performance by month using date_posted without modifying model features.
    """
    # Create copy and convert date_posted to datetime
    df = test_df.copy()
    df['date_posted'] = pd.to_datetime(df['date_posted'])
    
    # Extract month from date_posted
    date_series = df['date_posted']
    months = sorted(date_series.dt.to_period('M').unique())
    performances = []
    
    for month in months:
        # Create month mask
        month_mask = date_series.dt.to_period('M') == month
        month_data = df[month_mask]
        
        if len(month_data) > 0:
            if model_type != "baseline":
                # Drop date_posted and fully_funded for prediction
                X = month_data.drop(["not_fully_funded", "date_posted"], axis=1)
                y_true = month_data["not_fully_funded"]
                y_pred = classifier.predict(X)
            else:
                y_true = month_data["not_fully_funded"]
                y_pred = np.random.choice([0, 1], size=len(y_true))
            
            performance = recall_score(y_true, y_pred)
            performances.append((month, performance))
        else:
            performances.append((month, None))
    
    return performances

def evaluate_all_models_temporal(test_df, models, pov_lvl):
    """
    Evaluate all models over time and calculate regrets based on months.
    """
    # Collect performances for all models
    model_performances = {}
    months_set = set()
    
    # Collect months and performances
    for model_type in models:
        if model_type != "baseline":
            classifier = joblib.load(f"../outputs/{model_type}_{pov_lvl}_poverty.pkl")
        else:
            classifier = None
        
        performances = collect_temporal_performance(test_df, classifier, model_type)
        model_performances[model_type] = performances
        months_set.update(month for month, _ in performances)
    
    months = sorted(months_set)
    
    aligned_performances = {}
    for model_type, perfs in model_performances.items():
        perf_dict = dict(perfs)
        aligned_performances[model_type] = [perf_dict.get(month) for month in months]
    
    # Calculate best performance for each month
    best_performance_by_period = []
    for i in range(len(months)):
        period_performances = [
            perfs[i] for perfs in aligned_performances.values() if perfs[i] is not None
        ]
        if period_performances:
            best_performance_by_period.append(max(period_performances))
        else:
            best_performance_by_period.append(0)
    
    regrets = calculate_temporal_regret(aligned_performances, best_performance_by_period)
    
    for model_name in regrets:
        regrets[model_name]['months'] = months
    
    # Plot regret over time
    plot_regret_over_time(regrets, models)
    
    return regrets, months

def plot_regret_over_time(regrets, model_names):
    """
    Visualize regret trends over time with month labels.
    """
    plt.figure(figsize=(12, 6))
    
    # Get months from the regrets dictionary
    months = regrets[next(iter(regrets))]['months']
    month_labels = [str(month.strftime('%b %Y')) for month in months]
    
    for model in model_names:
        regret_values = regrets[model]['regret_by_period']
        plt.plot(range(len(months)), regret_values, marker='o', label=model, linewidth=2)
    
    plt.xlabel('Month')
    plt.ylabel('Regret')
    plt.title('Model Regret Over Time (by Month)')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(len(months)), month_labels, rotation=45)
    plt.tight_layout()
    plt.savefig("../figures/regret_over_time_by_month.jpg")
    plt.clf()

def calculate_temporal_regret(model_performances, best_model_performance):
    """
    Calculate regret across time periods (months) for model selection.
    """
    regrets = {}
    for model_name, performances in model_performances.items():
        model_regrets = []
        for period, perf in enumerate(performances):
            if perf is not None:
                regret = best_model_performance[period] - perf
            else:
                regret = 0
            model_regrets.append(regret)
        
        regrets[model_name] = {
            'mean_regret': np.mean([r for r in model_regrets if r is not None]),
            'max_regret': max([r for r in model_regrets if r is not None], default=0),
            'regret_variance': np.var([r for r in model_regrets if r is not None], ddof=1) 
                               if len(model_regrets) > 1 else 0,
            'regret_by_period': model_regrets
        }
    
    return regrets

def print_regret_summary(regrets, months, pov_lvl=""):
    """
    Print formatted summary of regret metrics with month information.
    """
    print(f"\nRegret Summary{' for ' + pov_lvl if pov_lvl else ''}:")
    summary_data = []
    
    for model_name, metrics in regrets.items():
        summary_data.append({
            'Model': model_name,
            'Mean Regret': metrics['mean_regret'],
            'Max Regret': metrics['max_regret'],
            'Regret Variance': metrics['regret_variance'],
            'Months Evaluated': f"{months[0].strftime('%b %Y')} - {months[-1].strftime('%b %Y')}"
        })
    
    # Convert to DataFrame for formatting
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv(f"../outputs/regret_summary{('_' + pov_lvl) if pov_lvl else ''}_by_month.csv", 
                      index=False)
    
def plot_model_bias(model_type, metric, protected_groups, protected_group_names, output, test_df, bias_categories, pov_lvl):
    # set up plot
    y = np.arange(len(bias_categories))
    plt.figure(figsize=(6, 4)) 

    all_scores = []
    for i in range(len(protected_groups)):
        grp = protected_groups[i]
        name = protected_group_names[i]
        
        test_df["is_protected"] = test_df[grp].any(axis=1)
        non_protected_cols = [col for col in test_df.columns if name in col and col not in grp]
        test_df["not_protected"] = test_df[non_protected_cols].any(axis=1)
        ref_mask = test_df["not_protected"] == True
        protect_mask = test_df["is_protected"] == True

        if metric == "recall":
            score_ref = recall_score(output["y_test"][ref_mask], output["y_pred"][ref_mask])
            score_protect = recall_score(output["y_test"][protect_mask], output["y_pred"][protect_mask])

        if metric == "precision":
            score_ref = recall_score(output["y_test"][ref_mask], output["y_pred"][ref_mask])
            score_protect = recall_score(output["y_test"][protect_mask], output["y_pred"][protect_mask])

        all_scores.append(score_ref)
        all_scores.append(score_protect)

    plt.barh(y, all_scores, color=['lightgreen', 'lightblue', 'lightgreen', 'lightblue'])
    plt.yticks(y, bias_categories)
    plt.xlabel('Recall')
    plt.title(f"{model_type} {pov_lvl} poverty level")
    plt.tight_layout()
    
     # Save the plot
    plt.savefig(f"../figures/{pov_lvl}_{model_type}_{metric}_bias_plot.jpg")
    plt.clf()


if __name__ == "__main__":
    config = load_config()
    models = config["models"]

    split_by_poverty = config["split_by_poverty"]
    stem_cols = config["stem_cols"]
    tech_cols = config["tech_cols"]
    test_results = {}
    all_results = {}

    test_path = "../outputs/test_df.csv"
    test_df = pd.read_csv(test_path)
    test_df = test_df.set_index("projectid")

    pov_lvl = "none"
    classifier = ""

    print("\nCalculating overall temporal regret metrics...")
    regrets, years = evaluate_all_models_temporal(test_df, models, "none")
    print_regret_summary(regrets, years)
    
    # Overall evaluation
    all_results['none'] = {}
    for model_type in models:
        if model_type != "baseline":
            classifier = joblib.load("../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")
        
        y, y_pred, y_pred_probs, X = evaluate(test_df, classifier, model_type, "none")
        test_results[model_type + pov_lvl] = {"y_test": y, "y_pred": y_pred}
        
        # Store metrics
        all_results['none'][model_type] = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred)
        }

    plot_false_discovery_rate_ref(models, test_results, X, stem_cols, "primary_focus_subject", pov_lvl)
    plot_recall_disparity(models, test_results, X, stem_cols, "primary_focus_subject", pov_lvl)   

    plot_false_discovery_rate_ref(models, test_results, X, tech_cols, "resource_type", pov_lvl)
    plot_recall_disparity(models, test_results, X, tech_cols, "resource_type", pov_lvl)

    # Poverty level specific evaluation
    if split_by_poverty == "true":
        for pov_lvl, pov_col_name in config["poverty_columns"].items():
            all_results[pov_lvl] = {}
            test_results = {}
            
            for model_type in models:
                if model_type != "baseline":
                    classifier = joblib.load("../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")
                
                pov_test_path = f"../outputs/{pov_lvl}_pov_lvl_test_df.csv"
                pov_test_df = pd.read_csv(pov_test_path)
                pov_test_df = pov_test_df.set_index("projectid")
                
                y, y_pred, y_pred_probs, X = evaluate(pov_test_df, classifier, model_type, pov_lvl)
                test_results[model_type + pov_lvl] = {"y_test": y, "y_pred": y_pred}
                
                # Store metrics
                all_results[pov_lvl][model_type] = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred),
                    'recall': recall_score(y, y_pred)
                }

                if model_type == "svm":
                    plot_model_bias(model_type, "recall", [stem_cols, tech_cols], ["primary_focus_subject", "resource_type"], test_results[model_type + pov_lvl], pov_test_df, ["non stem", "stem", "non tech", "tech"], pov_lvl)
                    plot_model_bias(model_type, "precision", [stem_cols, tech_cols], ["primary_focus_subject", "resource_type"], test_results[model_type + pov_lvl], pov_test_df, ["non stem", "stem", "non tech", "tech"], pov_lvl)
    
            plot_false_discovery_rate_ref(models, test_results, X, stem_cols, "primary_focus_subject", pov_lvl)
            plot_recall_disparity(models, test_results, X, stem_cols, "primary_focus_subject", pov_lvl)   

            plot_false_discovery_rate_ref(models, test_results, X, tech_cols, "resource_type", pov_lvl)
            plot_recall_disparity(models, test_results, X, tech_cols, "resource_type", pov_lvl)  

    print_summary_table(all_results)
    print(f"Model evaluation complete.")

