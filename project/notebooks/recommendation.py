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

# Load test data
test_df = pd.read_csv('../outputs/test_df.csv')
test_df['date_posted'] = pd.to_datetime(test_df['date_posted'])

# Filter projects posted for at least 3 months
fixed_max_date = pd.to_datetime('2013-12-31')
test_df['months_since_posted'] = ((fixed_max_date - test_df['date_posted']) / pd.Timedelta(days=30)).astype(int)
eligible_projects = test_df[test_df['months_since_posted'] >= 3].copy()

# Load config and initialize parameters
config = load_config()
poverty_levels = ["low", "moderate", "high", "highest"]
models = ["random_forest", "logistic_regression"]
recommendations = {}

for model_type in models:
    for pov_level in poverty_levels:
        pov_column = f"poverty_level_{pov_level} poverty"
        pov_projects = eligible_projects[eligible_projects[pov_column] == 1].copy()

        # Load the classifier
        classifier = joblib.load(f"../outputs/{model_type}_{pov_level}_poverty.pkl")

        # Prepare the data
        X_test_filtered = pov_projects.drop(columns=['fully_funded', 'date_posted', 'months_since_posted', 'projectid'], errors='ignore')
        X_test_filtered = X_test_filtered.reindex(columns=classifier.feature_names_in_, fill_value=0)

        # Predict probability of being fully funded
        pov_projects[f'probability_fully_funded_{model_type}'] = classifier.predict_proba(X_test_filtered)[:, 1]
        pov_projects_sorted = pov_projects.sort_values(by=f'probability_fully_funded_{model_type}', ascending=False)

        # Select top recommendations based on poverty level
        if pov_level in ["high", "highest"]:
            top_recommendations = pov_projects_sorted.head(10)
        elif pov_level in ["low", "moderate"]:
            top_recommendations = pov_projects_sorted.head(3)

        # Store recommendations
        recommendations[(model_type, pov_level)] = top_recommendations[['projectid', 'date_posted', f'probability_fully_funded_{model_type}']]

# Create heatmaps for each model and poverty level combination
for key, recs in recommendations.items():
    model_type, pov_level = key

    # Pivot data for the heatmap
    heatmap_data = recs.pivot_table(
        index='projectid',
        values=f'probability_fully_funded_{model_type}',
        aggfunc='mean'
    )

    # Plot heatmap
    plt.figure(figsize=(8, 10))
    sns.heatmap(
        heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f",
        cbar_kws={'label': 'Probability of Fully Funded'}
    )
    plt.title(f"{model_type.capitalize()} - {pov_level.capitalize()} Poverty Level")
    plt.xlabel("Model")
    plt.ylabel("Project ID")

    plt.savefig("../outputs/{model_type}_{pov_level}_poverty_level_heatmap.png", dpi=300, bbox_inches="tight")
