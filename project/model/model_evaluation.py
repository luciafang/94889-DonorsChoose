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
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_type} ROC Curve with Time Series Cross-Validation")
    plt.legend(loc='lower right')
    plt.savefig("../figures/" + model_type  + "_" + pov_lvl +  "_roc.jpg")
    plt.clf()

def metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

#Find precision and recall (true values, scores, k_increase)
def pre_rec(y_t, y_s, model_type, pov_lvl, k_inc=0.01):
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
    plt.savefig("../figures/" + model_type + "_" + pov_lvl + "_prk.jpg")
    plt.clf()

    return (precision_k,recall_k)

def baseline_predict(df, sort_col):
    copy_df = df.copy()

    # use specified column sort_col to sort dataset and predict outcome variables
    sorted_df = copy_df.sort_values(by=sort_col, ascending=False)

    # get values in column
    sort_col_df = sorted_df[sort_col].to_frame()

    # use .3 bc 30% of projects don't get funded, so want to predict top 30 in sort_col as not getting funded
    split_val = round(.3 * len(sort_col_df))

    copy_df["fully_funded_pred"] = 0
    
    # predict all the projects with a value less than the split_val as getting funded, should be about 70% of projects
    copy_df.loc[copy_df[sort_col] < sort_col_df.iloc[split_val].values[0], "fully_funded_pred"] = 1

    # return y_test, y_pred
    return copy_df["fully_funded_pred"]

def evaluate(test_df, classifier, model_type, pov_lvl):
    print(model_type)
    X = test_df.copy()
    y = test_df["fully_funded"]
    if "fully_funded" in test_df.columns:
        X = X.drop(["fully_funded"] , axis=1)
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
    metrics(y, y_pred)
    return y, y_pred, y_pred_probs, X

def plot_false_discovery_rate_ref(model_names, model_outputs, test_df, stem_cols, pov_lvl):
    test_df["is_stem"] = test_df[stem_cols].any(axis=1)
    non_stem = [col for col in test_df.columns if "primary_focus_subject" in col and col not in stem_cols]
    test_df["not_stem"] = test_df[non_stem].any(axis=1)
    ref_mask = test_df["not_stem"] == True
    protect_mask = test_df["is_stem"] == True
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
    plt.title('FDR Disparity vs Precision')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1.5)

    # Save the plot
    plt.savefig("../figures/" + pov_lvl + "_fdr_disparity_plot.jpg")
    plt.clf()

def plot_recall_disparity(model_names, model_outputs, test_df, stem_cols, pov_lvl):
    test_df["is_stem"] = test_df[stem_cols].any(axis=1)
    non_stem = [col for col in test_df.columns if "primary_focus_subject" in col and col not in stem_cols]
    test_df["not_stem"] = test_df[non_stem].any(axis=1)
    ref_mask = test_df["not_stem"] == True
    protect_mask = test_df["is_stem"] == True

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
    plt.title('Recall Disparity vs Precision')
    plt.legend()
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)

    # Save the plot
    plt.savefig("../figures/" + pov_lvl + "_recall_disparity_plot.jpg")
    plt.clf()

def collect_temporal_performance(test_df, classifier, model_type):
    """
    Collect model performance by year using date_posted without modifying model features
    """
    # Create copy and convert date_posted to datetime
    df = test_df.copy()
    
    # Extract year from date_posted without adding it as a feature
    date_series = pd.to_datetime(df['date_posted'])
    years = sorted(date_series.dt.year.unique())
    performances = []
    
    for year in years:
        # Create year mask without modifying the dataframe
        year_mask = date_series.dt.year == year
        year_data = df[year_mask]
        
        if len(year_data) > 0:
            if model_type != "baseline":
                # Drop date_posted and fully_funded for prediction
                X = year_data.drop(["fully_funded", "date_posted"], axis=1)
                y_true = year_data["fully_funded"]
                y_pred = classifier.predict(X)
            else:
                X = year_data.copy()
                y_true = year_data["fully_funded"]
                y_pred = baseline_predict(X, "total_price_excluding_optional_support")
            
            perf = precision_score(y_true, y_pred)
            performances.append((year, perf))
    
    return performances

def evaluate_all_models_temporal(test_df, models, pov_lvl):
    """
    Evaluate all models over time and calculate regrets
    """
    # Collect performances for all models
    model_performances = {}
    years_set = set()
    
    # First pass to collect all years and performances
    for model_type in models:
        if model_type != "baseline":
            classifier = joblib.load("../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")
        else:
            classifier = None
            
        performances = collect_temporal_performance(test_df, classifier, model_type)
        model_performances[model_type] = performances
        years_set.update(year for year, _ in performances)
    
    # Sort years for consistent ordering
    years = sorted(years_set)
    
    # Create aligned performance lists
    aligned_performances = {}
    for model_type, perfs in model_performances.items():
        perf_dict = dict(perfs)
        aligned_performances[model_type] = [perf_dict.get(year) for year in years]
    
    # Calculate best performance for each year
    best_performance_by_period = []
    for i in range(len(years)):
        period_performances = [perfs[i] for perfs in aligned_performances.values() 
                             if perfs[i] is not None]
        if period_performances:  # Check if there are any performances for this period
            best_performance_by_period.append(max(period_performances))
        else:
            best_performance_by_period.append(0)  # Or some other appropriate default
    
    # Calculate regrets
    regrets = calculate_temporal_regret(aligned_performances, best_performance_by_period)
    
    # Add years information to regrets for plotting
    for model_name in regrets:
        regrets[model_name]['years'] = years
    
    # Plot regret over time
    plot_regret_over_time(regrets, models)
    
    return regrets, years

def plot_regret_over_time(regrets, model_names):
    """
    Visualize regret trends over time with year labels
    """
    plt.figure(figsize=(10, 6))
    
    # Get years from first model (they're all the same)
    years = regrets[next(iter(regrets))]['years']
    
    for model in model_names:
        regret_values = regrets[model]['regret_by_period']
        plt.plot(range(len(years)), regret_values, marker='o', label=model, linewidth=2)
    
    plt.xlabel('Year')
    plt.ylabel('Regret')
    plt.title('Model Regret Over Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(len(years)), years, rotation=45)
    plt.tight_layout()
    plt.savefig("../figures/regret_over_time.jpg")
    plt.clf()

def calculate_temporal_regret(model_performances, best_model_performance):
    """
    Calculate regret across time periods for model selection
    """
    regrets = {}
    for model_name, performances in model_performances.items():
        model_regrets = []
        for period, perf in enumerate(performances):
            if perf is not None and best_model_performance[period] is not None:
                regret = best_model_performance[period] - perf
            else:
                regret = 0  # Or some other appropriate default value
            model_regrets.append(regret)
        
        regrets[model_name] = {
            'mean_regret': np.mean([r for r in model_regrets if r is not None]),
            'max_regret': max([r for r in model_regrets if r is not None], default=0),
            'regret_variance': np.var([r for r in model_regrets if r is not None]) if any(r is not None for r in model_regrets) else 0,
            'regret_by_period': model_regrets
        }
    
    return regrets

def print_regret_summary(regrets, years, pov_lvl=""):
    """
    Print formatted summary of regret metrics with year information
    """
    print(f"\nRegret Summary{' for ' + pov_lvl if pov_lvl else ''}:")
    summary_data = []
    
    for model_name, metrics in regrets.items():
        summary_data.append({
            'Model': model_name,
            'Mean Regret': metrics['mean_regret'],
            'Max Regret': metrics['max_regret'],
            'Regret Variance': metrics['regret_variance'],
            'Years Evaluated': f"{min(years)}-{max(years)}"
        })
    
    # Convert to DataFrame for nice formatting
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv(f"../outputs/regret_summary{('_' + pov_lvl) if pov_lvl else ''}.csv", 
                     index=False)

if __name__ == "__main__":
    config = load_config()
    models = config["models"]

    split_by_poverty = config["split_by_poverty"]
    stem_cols = config["stem_cols"]
    test_results = {}

    test_path = "../outputs/test_df.csv"
    test_df = pd.read_csv(test_path)
    test_df = test_df.set_index("projectid")

    pov_lvl = "none"
    classifier = ""

    print("\nCalculating overall temporal regret metrics...")
    regrets, years = evaluate_all_models_temporal(test_df, models, "none")
    print_regret_summary(regrets, years)
    
    for model_type in models:
        if model_type != "baseline":
            classifier = joblib.load("../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")

        print(pov_lvl + " poverty level")
        y, y_pred, y_pred_probs, X = evaluate(test_df, classifier, model_type, pov_lvl)
        test_results[model_type + pov_lvl] = {"y_test": y, "y_pred": y_pred}
    plot_false_discovery_rate_ref(models, test_results, X, stem_cols, pov_lvl)
    plot_recall_disparity(models, test_results, X, stem_cols, pov_lvl)   
    print("#######################")
    if split_by_poverty == "true":
        for pov_lvl, pov_col_name in config["poverty_columns"].items():
            test_results = {}
            classifier = ""
            for model_type in models:
                if model_type != "baseline":
                    classifier = joblib.load("../outputs/" + model_type + f"_{pov_lvl}_poverty.pkl")
                print(pov_lvl + " poverty level")
                smote_test_path = f"../outputs/{pov_lvl}_pov_lvl_test_df.csv"
                smote_test_df = pd.read_csv(smote_test_path)

                y, y_pred, y_pred_probs, X = evaluate(smote_test_df, classifier, model_type, pov_lvl)
                test_results[model_type + pov_lvl] = {"y_test": y, "y_pred": y_pred}
            plot_false_discovery_rate_ref(models, test_results, X, stem_cols, pov_lvl)
            plot_recall_disparity(models, test_results, X, stem_cols, pov_lvl)  
            print("#######################")
            # test_results[model_type + pov_lvl] = {"y_test": y, "y_pred": y_pred}

     

    print(f"Model evaluation complete.")

