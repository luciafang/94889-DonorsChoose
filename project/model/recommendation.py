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


if __name__ == "__main__":
    # Load test data
    test_df = pd.read_csv('../outputs/test_df.csv')
    test_df['date_posted'] = pd.to_datetime(test_df['date_posted'])
    test_df['projectid'] = test_df.index

    fixed_max_date = pd.to_datetime('2013-12-31')
    test_df['months_since_posted'] = ((fixed_max_date - test_df['date_posted']) / pd.Timedelta(days=30)).astype(int)
    eligible_projects = test_df[test_df['months_since_posted'] >= 3].copy()

    config = load_config()
    poverty_levels = ["low", "high"]
    models = ["random_forest", "logistic_regression"]
    recommendations = {}
    all_recommendations = eligible_projects.copy()

    for model_type in models:
        for pov_level in poverty_levels:
            pov_column = f"poverty_level_{pov_level} poverty"
            pov_projects = eligible_projects[eligible_projects[pov_column] == 1].copy()

            classifier = joblib.load(f"../outputs/{model_type}_{pov_level}_poverty.pkl")

            X_test_filtered = pov_projects.drop(
                columns=['fully_funded', 'date_posted', 'months_since_posted', 'projectid'], errors='ignore')
            X_test_filtered = X_test_filtered.reindex(columns=classifier.feature_names_in_, fill_value=0)

            proba_column = f'proba_{model_type}'
            pov_projects[proba_column] = classifier.predict_proba(X_test_filtered)[:, 1]
            all_recommendations.loc[pov_projects.index, proba_column] = pov_projects[proba_column]

    top_recommendations = {}
    for pov_level in poverty_levels:
        pov_column = f"poverty_level_{pov_level} poverty"
        pov_projects = all_recommendations[all_recommendations[pov_column] == 1]

        top_n = 10 if pov_level in ["high", "highest"] else 3

        top_rf = pov_projects.nlargest(top_n, 'proba_random_forest').copy()
        top_recommendations[(pov_level, 'random_forest')] = top_rf
        top_logistic = pov_projects.nlargest(top_n, 'proba_logistic_regression').copy()
        top_recommendations[(pov_level, 'logistic_regression')] = top_logistic

        top_rf.to_csv(f"../outputs/random_forest_{pov_level}_top_recommendations.csv", index=False)
        top_logistic.to_csv(f"../outputs/logistic_regression_{pov_level}_top_recommendations.csv", index=False)
        print(f"Saved top recommendations for {model_type} and {pov_level}")

    feature_categories = {
        'STEM': [
            'primary_focus_subject_Applied Sciences',
            'primary_focus_subject_Environmental Science',
            'primary_focus_subject_Health & Life Science',
            'primary_focus_subject_Mathematics',
            'primary_focus_subject_Nutrition',
            'primary_focus_subject_Health & Wellness'
        ],
        'Non-STEM': [
            col for col in all_recommendations.columns if col.startswith('primary_focus_subject_')
                                                          and col not in [
                                                              'primary_focus_subject_Applied Sciences',
                                                              'primary_focus_subject_Environmental Science',
                                                              'primary_focus_subject_Health & Life Science',
                                                              'primary_focus_subject_Mathematics',
                                                              'primary_focus_subject_Nutrition',
                                                              'primary_focus_subject_Health & Wellness'
                                                          ]
        ],
        'Resource Type': [col for col in all_recommendations.columns if col.startswith('resource_type_')],
    }

    poverty_levels = ["low", "high"]

    heatmap_data = []

    for pov_level in poverty_levels:
        for category, features in feature_categories.items():
            rf_log_diff = (top_recommendations[(pov_level, 'random_forest')][features].sum() -
                           top_recommendations[(pov_level, 'logistic_regression')][features].sum())
            # pov_data = all_recommendations[all_recommendations[f'poverty_level_{pov_level} poverty'] == 1]
            # rf_log_diff = pov_data['proba_random_forest'] - pov_data['proba_logistic_regression']

            sum_diff = rf_log_diff.sum()  # Mean difference in funding probability for this category

            heatmap_data.append({
                'poverty_level': pov_level,
                'feature_category': category,
                'total_difference': sum_diff
            })

    heatmap_df = pd.DataFrame(heatmap_data)

    poverty_level_order = ["high", "low"]
    feature_category_order = ["STEM", "Non-STEM", "Resource Type"]
    heatmap_pivot = heatmap_df.pivot(index="feature_category", columns="poverty_level", values="total_difference")
    heatmap_pivot = heatmap_pivot.reindex(index=feature_category_order, columns=poverty_level_order)

    plt.figure(figsize=(5, 4))
    sns.heatmap(heatmap_pivot,
                # cmap = 'rocket',
                cmap='RdBu_r',
                cbar=False,
                annot=True, center=0)
    plt.title('Projects recommendation differences RF - LR')
    plt.xlabel('Poverty Level')
    plt.ylabel('Feature Category')
    plt.savefig('../figures/recommendation_difference_heatmap.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
