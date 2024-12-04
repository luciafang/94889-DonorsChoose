import pandas as pd
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt
from model_evaluation import metrics


def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def get_recommended_projects(projects_around_3_months, model_type, pov_level, quant_variables):
    """
    gets the projects to recommend from the model output 

    projects_around_3_months: dataframe of output of model + the project info + prediction 
    probabilities for the projects elgible for recommendation
    model_type: the string name of the type of model
    pov_level: the string name of the poverty level of the model output
    quant_variables: the names of the quantitative variables
    """
    if "svm" in model_type:
        classifier = joblib.load("../outputs/" + model_type + f"_calibrated_{pov_level}_poverty.pkl")
    else:
        classifier = joblib.load(f"../outputs/{model_type}_{pov_level}_poverty.pkl")
    X_test = projects_around_3_months.drop(['not_fully_funded', 'date_posted', 'months_posted', 'projectid'], axis=1)
    pred_probs = classifier.predict_proba(X_test)[:, 1]
    preds = classifier.predict(X_test)
    projects_around_3_months_wtih_probs = projects_around_3_months.copy()
    projects_around_3_months_wtih_probs["pred_prob"] = pred_probs
    projects_around_3_months_wtih_probs["pred"] = preds

    # unscale data to see original values
    scaler_path = f"../outputs/{pov_level}_poverty_level_scaler.pkl"
    scaler = joblib.load(scaler_path)
    
    # want to only recommend projects with at least $100 price
    projects_around_3_months_wtih_probs[quant_variables] = scaler.inverse_transform(projects_around_3_months_wtih_probs[quant_variables])
    projects_around_3_months_wtih_probs = projects_around_3_months_wtih_probs[projects_around_3_months_wtih_probs["total_price_excluding_optional_support"] > 100]

    # want to only recommend projects that have at least 50% funding HEREIN LIES THE PROBLEM
    projects_around_3_months_wtih_probs = projects_around_3_months_wtih_probs[projects_around_3_months_wtih_probs["percentage_reached_month_3"] > 20]

    # sort model output by prediction probability
    projects_with_impact = projects_around_3_months_wtih_probs.copy()
    projects_with_impact["expected_impact"] = projects_around_3_months_wtih_probs["pred_prob"] * projects_around_3_months_wtih_probs["students_reached"]
    sorted_output = projects_with_impact.sort_values('expected_impact', ascending=False)

    # take top number of recommendations
    if pov_level == "low":
        # sort output by oldest posted date
        top_recs = sorted_output[:100].sort_values('date_posted', ascending=True)
    elif pov_level == "high":
        # sort output by oldest posted date
        top_recs = sorted_output[:200].sort_values('date_posted', ascending=True)
    output_path = f"../outputs/{model_type}_{pov_level}_pov_lvl_final_model_output.csv"
    top_recs.to_csv(output_path)
    
    # return top number of recommendations
    return top_recs

def get_recommended_projects_baseline(projects_around_3_months, model_type, pov_level):
    """
    gets the projects to recommend from the model output for the baseline model

    projects_around_3_months: dataframe of output of model + the project info + prediction 
    probabilities for the projects elgible for recommendation
    model_type: the string name of the type of model
    pov_level: the string name of the poverty level of the model output
    """
    sorted_by_price = projects_around_3_months.sort_values('total_price_excluding_optional_support', ascending=False)
    top_recs = pd.DataFrame()
    if pov_level == "low":
        # sort output by oldest posted date
        top_recs = sorted_by_price[:100]
    elif pov_level == "high":
        # sort output by oldest posted date
        top_recs = sorted_by_price[:200]
    
    # get values in column
    price_df = top_recs['total_price_excluding_optional_support'].to_frame()

    # use .3 bc 30% of projects don't get funded, so want to predict top 30 in sort_col as not getting funded
    split_val = round(.3 * len(price_df))

    top_recs["pred"] = 0
    
    # predict all the projects with a value less than the split_val as getting funded, should be about 70% of projects
    top_recs.loc[top_recs['total_price_excluding_optional_support'] < price_df.iloc[split_val].values[0], "pred"] = 1

    top_recs = top_recs.sort_values('date_posted', ascending=True)

    output_path = f"../outputs/{model_type}_{pov_level}_pov_lvl_final_model_output.csv"
    top_recs.to_csv(output_path)
    
    # return top number of recommendations
    return top_recs

def get_projects_at_3_months(date_for_recommendation, df):
    """
    gets the projects posted around 3 months before the date the recommendation is being made
    
    date_for_recommendation: date to make recommendation on for the month
    df: dataframe to get the projects posted around 3 months prior to recommendation date
    """
    # get all projects posted before recommendation date
    projects_before_rec_date = df[df['date_posted'] <= date_for_recommendation]
    # add column for how long projects posted
    projects_before_rec_date['months_posted'] = round(((date_for_recommendation - projects_before_rec_date['date_posted']) / pd.Timedelta(days=30)))
    # want projects posted at least 3 months
    projects_around_3_months = projects_before_rec_date[projects_before_rec_date['months_posted'] == 3]
    return projects_around_3_months

def eval_recommendation(y_pred, y_test):
    metrics(y_test, y_pred)

if __name__ == "__main__":
    config = load_config()

    models = config["models"]
    poverty_levels = config["poverty_columns"]
    quant_variables = config["quant_variables"]

    for pov_level in poverty_levels:
        # Load test data
        test_path = f"../outputs/{pov_level}_pov_lvl_test_df.csv"
        test_df = pd.read_csv(test_path)
        test_df['date_posted'] = pd.to_datetime(test_df['date_posted'])

        # can make first recommendation on April 1st, 2013
        date_for_recommendation = pd.to_datetime('2013-04-01')

        projects_around_3_months = get_projects_at_3_months(date_for_recommendation, test_df)
        all_recs = {}
        for model_type in models:
            if model_type != "baseline":
                recs = get_recommended_projects(projects_around_3_months, model_type, pov_level, quant_variables)
            else:
                recs = get_recommended_projects_baseline(projects_around_3_months, model_type, pov_level)
            all_recs[f"{model_type}_{pov_level}"] = recs
            print(f"Metrics for {pov_level} poverty level and {model_type}")
            y_pred = recs["pred"]
            y_test = recs["not_fully_funded"]
            eval_recommendation(y_pred, y_test)
            print(f"Saved recommendation model output for {model_type} and {pov_level} poverty level")
