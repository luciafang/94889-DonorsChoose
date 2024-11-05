import pandas as pd
import numpy as np
import json
import os

def load_config(config_file="../config.json"):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def percent_reached(df_projects, df_donations):
    """
    Calculate the percentage of project funding reached over time.
    """
    # Convert date columns to datetime
    df_projects['date_posted'] = pd.to_datetime(df_projects['date_posted'], errors='coerce')
    df_donations['donation_timestamp'] = pd.to_datetime(df_donations['donation_timestamp'], errors='coerce')

    # Extract the month from 'date_posted' and create a new column 'month_posted'
    df_projects['month_posted'] = df_projects['date_posted'].dt.month_name()

    # Merge donations with projects
    merged_data = df_donations.merge(
        df_projects[['projectid', 'total_price_excluding_optional_support', 'date_posted']],
        on='projectid', how='left'
    )

    # Calculate the time difference in months
    merged_data['donation_month'] = ((merged_data['donation_timestamp'] - merged_data['date_posted']).dt.days // 30) + 1

    # Filter donations within the first four months
    merged_data = merged_data[merged_data['donation_month'].between(1, 4)]

    # Group by project and month, then sum the donations
    monthly_donations = merged_data.groupby(['projectid', 'donation_month'])['donation_to_project'].sum().reset_index()

    # Pivot the table to get months as columns
    monthly_donations_pivot = monthly_donations.pivot(index='projectid', columns='donation_month', values='donation_to_project').fillna(0)
    monthly_donations_pivot.columns = [f'month_{int(col)}_donations' for col in monthly_donations_pivot.columns]

    # Calculate cumulative donations for each project up to each month
    monthly_donations_pivot['cumulative_donations_month_1'] = monthly_donations_pivot['month_1_donations']
    monthly_donations_pivot['cumulative_donations_month_2'] = monthly_donations_pivot['cumulative_donations_month_1'] + monthly_donations_pivot.get('month_2_donations', 0)
    monthly_donations_pivot['cumulative_donations_month_3'] = monthly_donations_pivot['cumulative_donations_month_2'] + monthly_donations_pivot.get('month_3_donations', 0)
    monthly_donations_pivot['cumulative_donations_month_4'] = monthly_donations_pivot['cumulative_donations_month_3'] + monthly_donations_pivot.get('month_4_donations', 0)

    # Merge back to projects to calculate cumulative percentages
    df_projects = df_projects.merge(monthly_donations_pivot, on='projectid', how='left')

    # Calculate cumulative percentage of donations reached for each month
    for month in range(1, 5):
        df_projects[f'percentage_reached_month_{month}'] = (
            df_projects[f'cumulative_donations_month_{month}'] / df_projects['total_price_excluding_optional_support']
        ).fillna(0) * 100

    # Fill NaN values with 0 for donation columns
    donation_cols = [f'month_{i}_donations' for i in range(1, 5)]
    df_projects[donation_cols] = df_projects[donation_cols].fillna(0)

    # Drop cumulative donation columns
    df_projects.drop(columns=[
        'cumulative_donations_month_1',
        'cumulative_donations_month_2',
        'cumulative_donations_month_3',
        'cumulative_donations_month_4'
    ], inplace=True)

    return df_projects

def aggregate_geographic(df_projects):
    """
    Aggregate projects by city and state.
    """
    # Total projects in city
    projects_by_city = df_projects.groupby('school_city').size().reset_index(name='total_projects_in_city')
    df_projects = df_projects.merge(projects_by_city, on='school_city', how='left')

    # Total projects in state
    projects_by_state = df_projects.groupby('school_state').size().reset_index(name='total_projects_in_state')
    df_projects = df_projects.merge(projects_by_state, on='school_state', how='left')

    return df_projects[['projectid', 'total_projects_in_city', 'total_projects_in_state']]

if __name__ == "__main__":
    # Load data and config
    config = load_config()
    donations_file = config["raw_datasets"]["donations"]
    projects_file = config["raw_datasets"]["projects"]
    cleaned_dataset_file = "../outputs/selected_dataset.csv"

    df_donations = pd.read_csv(donations_file)
    df_projects = pd.read_csv(projects_file)
    df_cleaned = pd.read_csv(cleaned_dataset_file)

    # Geographic aggregation features
    geographic_features = aggregate_geographic(df_projects)

    # Percentage reached feature engineering
    df_projects_percent = percent_reached(df_projects, df_donations)

    # Merge the new features into the cleaned dataset
    df_cleaned = df_cleaned.merge(geographic_features, on='projectid', how='left')
    df_cleaned = df_cleaned.merge(df_projects_percent[[
        'projectid',
        # 'month_1_donations',
        # 'month_2_donations',
        # 'month_3_donations',
        # 'month_4_donations',
        'month_posted',
        'percentage_reached_month_1',
        'percentage_reached_month_2',
        'percentage_reached_month_3',
        'percentage_reached_month_4'
    ]], on='projectid', how='left')

    # Save the updated dataset with new features
    cleaned_output_path = "../outputs/selected_dataset_with_new_features.csv"
    output_dir = os.path.dirname(cleaned_output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df_cleaned.to_csv(cleaned_output_path, index=False)

    print(f"Feature engineering complete. Updated dataset saved to {cleaned_output_path}.")