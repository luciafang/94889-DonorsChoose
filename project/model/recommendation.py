import pandas as pd
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt


def load_config(config_file="../config.json"):
    """Load configuration from a JSON file."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def get_recommended_projects(model_type, pov_lvl, quant_variables):
    """
    model_output: dataframe of output of model + the project info + prediction probabilities
    num_recommendations: number of projects to recommend from model_output
    """
    if "svm" in model_type:
        classifier = joblib.load("../outputs/" + model_type + f"_calibrated_{pov_level}_poverty.pkl")
    else:
        classifier = joblib.load(f"../outputs/{model_type}_{pov_level}_poverty.pkl")
    X_test = projects_around_3_months.drop(['fully_funded', 'date_posted', 'months_posted', 'projectid'], axis=1)
    pred_probs = classifier.predict_proba(X_test)[:, 1]
    preds = classifier.predict(X_test)
    projects_around_3_months_wtih_probs = projects_around_3_months.copy()
    projects_around_3_months_wtih_probs["pred_prob"] = pred_probs
    projects_around_3_months_wtih_probs["pred"] = preds

    # unscale data to see original values
    scaler_path = f"../outputs/{pov_lvl}_poverty_level_scaler.pkl"
    scaler = joblib.load(scaler_path)
    # want to only recommend projects with at least $100
    projects_around_3_months_wtih_probs[quant_variables] = scaler.inverse_transform(projects_around_3_months_wtih_probs[quant_variables])
    projects_around_3_months_wtih_probs = projects_around_3_months_wtih_probs[projects_around_3_months_wtih_probs["total_price_excluding_optional_support"] > 100]
    # sort model output by prediction probability
    sorted_output = projects_around_3_months_wtih_probs.sort_values('pred_prob')
    # take top number of recommendations
    if pov_level == "low":
        top_recs = sorted_output[:100]
    elif pov_level == "high":
        top_recs = sorted_output[:200]
    output_path = f"../outputs/{model_type}_{pov_level}_pov_lvl_final_model_output.csv"
    top_recs.to_csv(output_path)
    
    # return top number of recommendations
    return top_recs

def get_projects_at_3_months(date_for_recommendation, df):
    """
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
                recs = get_recommended_projects(model_type, pov_level, quant_variables)
                all_recs[f"{model_type}_{pov_level}"] = recs
                print(f"Saved recommendation model output for {model_type} and {pov_level} poverty level")
    # feature_categories = {
    #     'STEM': [
    #         'primary_focus_subject_Applied Sciences',
    #         'primary_focus_subject_Environmental Science',
    #         'primary_focus_subject_Health & Life Science',
    #         'primary_focus_subject_Mathematics',
    #         'primary_focus_subject_Nutrition',
    #         'primary_focus_subject_Health & Wellness'
    #     ],
    #     'Non-STEM': [
    #         col for col in all_recommendations.columns if col.startswith('primary_focus_subject_')
    #                                                       and col not in [
    #                                                           'primary_focus_subject_Applied Sciences',
    #                                                           'primary_focus_subject_Environmental Science',
    #                                                           'primary_focus_subject_Health & Life Science',
    #                                                           'primary_focus_subject_Mathematics',
    #                                                           'primary_focus_subject_Nutrition',
    #                                                           'primary_focus_subject_Health & Wellness'
    #                                                       ]
    #     ],
    #     'Resource Type': [col for col in all_recommendations.columns if col.startswith('resource_type_')],
    # }

    # poverty_levels = ["low", "high"]

    # heatmap_data = []

    # for pov_level in poverty_levels:
    #     for category, features in feature_categories.items():
    #         rf_log_diff = (top_recommendations[(pov_level, 'random_forest')][features].sum() -
    #                        top_recommendations[(pov_level, 'logistic_regression')][features].sum())
    #         # pov_data = all_recommendations[all_recommendations[f'poverty_level_{pov_level} poverty'] == 1]
    #         # rf_log_diff = pov_data['proba_random_forest'] - pov_data['proba_logistic_regression']

    #         sum_diff = rf_log_diff.sum()  # Mean difference in funding probability for this category

    #         heatmap_data.append({
    #             'poverty_level': pov_level,
    #             'feature_category': category,
    #             'total_difference': sum_diff
    #         })

    # heatmap_df = pd.DataFrame(heatmap_data)

    # poverty_level_order = ["high", "low"]
    # feature_category_order = ["STEM", "Non-STEM", "Resource Type"]
    # heatmap_pivot = heatmap_df.pivot(index="feature_category", columns="poverty_level", values="total_difference")
    # heatmap_pivot = heatmap_pivot.reindex(index=feature_category_order, columns=poverty_level_order)

    # plt.figure(figsize=(5, 4))
    # sns.heatmap(heatmap_pivot,
    #             # cmap = 'rocket',
    #             cmap='RdBu_r',
    #             cbar=False,
    #             annot=True, center=0)
    # plt.title('Projects recommendation differences RF - LR')
    # plt.xlabel('Poverty Level')
    # plt.ylabel('Feature Category')
    # plt.savefig('../outputs/recommendation_difference_heatmap.png', format='png', dpi=300, bbox_inches='tight')
    # plt.close()
