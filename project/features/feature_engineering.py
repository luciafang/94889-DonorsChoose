import pandas as pd
import numpy as np
import json

# Load configuration from JSON
def load_config(config_file="../config.json"):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def aggregate_temporal(df_donations):
    """
    Create temporal features such as avg donations per week, monthly total donations.
    """
    # Handle inconsistent datetime formats
    df_donations['donation_timestamp'] = pd.to_datetime(df_donations['donation_timestamp'], errors='coerce')

    # Remove rows where the timestamp couldn't be parsed
    df_donations = df_donations.dropna(subset=['donation_timestamp']).copy()  # Ensure it's a copy

    # Use .loc to assign new columns
    df_donations.loc[:, 'week'] = df_donations['donation_timestamp'].dt.isocalendar().week
    df_donations.loc[:, 'month'] = df_donations['donation_timestamp'].dt.month
    
    # Avg donations per week
    donations_weekly = df_donations.groupby(['projectid', 'week']).agg({
        'donation_to_project': 'sum'
    }).groupby('projectid').mean().rename(columns={'donation_to_project': 'avg_donations_per_week'})
    
    # Total donations per month
    donations_monthly = df_donations.groupby(['projectid', 'month']).agg({
        'donation_to_project': 'sum'
    }).groupby('projectid').mean().rename(columns={'donation_to_project': 'avg_donations_per_month'})
    
    return donations_weekly, donations_monthly

def aggregate_geographic(df_projects):
    """
    Aggregate projects by city and state while keeping projectid.
    Adds columns for total projects in that city and state.
    """
    # Count total projects by city
    projects_by_city = df_projects.groupby('school_city').size().reset_index(name='total_projects_in_city')
    df_city_with_counts = df_projects.merge(projects_by_city, on='school_city', how='left')
    projects_by_city = df_city_with_counts[['projectid', 'total_projects_in_city']]

    # Count total projects by state
    projects_by_state = df_projects.groupby('school_state').size().reset_index(name='total_projects_in_state')
    df_state_with_counts = df_projects.merge(projects_by_state, on='school_state', how='left')  # Corrected to merge on school_state
    projects_by_state = df_state_with_counts[['projectid', 'total_projects_in_state']]

    return projects_by_city, projects_by_state

def aggregate_donor(df_donations):
    """
    Create donor-based aggregation features.
    """
    # Avg donor contribution per project
    avg_donor_contribution = df_donations.groupby('projectid')['donation_to_project'].mean().reset_index(name='avg_donor_contribution')
    
    # Number of repeat donors per project
    repeat_donors = df_donations.groupby(['projectid', 'donor_acctid']).size().reset_index(name='donations_count')
    repeat_donors = repeat_donors[repeat_donors['donations_count'] > 1]
    repeat_donors_count = repeat_donors.groupby('projectid').size().reset_index(name='number_of_repeat_donors')
    
    return avg_donor_contribution, repeat_donors_count

if __name__ == "__main__":
    # Load data and config
    config = load_config()
    donations_file = config["raw_datasets"]["donations"]
    projects_file = config["raw_datasets"]["projects"]
    cleaned_dataset_file = "../outputs/selected_dataset.csv"
    
    df_donations = pd.read_csv(donations_file)
    df_projects = pd.read_csv(projects_file)
    df_cleaned = pd.read_csv(cleaned_dataset_file)
    
    # Temporal aggregation features
    donations_weekly, donations_monthly = aggregate_temporal(df_donations)

    # Donor aggregation features
    avg_donor_contribution, repeat_donors_count = aggregate_donor(df_donations)

    # Geographic aggregation features
    projects_by_city, projects_by_state = aggregate_geographic(df_projects)

    # removed these features because the test data (2014->) has no data for donations yet
    # the reason why these are the only ones removed is because the classifier were heavily relying on them
    # Merge the new features into the cleaned dataset (on 'projectid')
    # df_cleaned = df_cleaned.merge(donations_weekly, on='projectid', how='left')
    # df_cleaned = df_cleaned.merge(donations_monthly, on='projectid', how='left')
    # df_cleaned = df_cleaned.merge(avg_donor_contribution, on='projectid', how='left')
    # df_cleaned = df_cleaned.merge(repeat_donors_count, on='projectid', how='left')
    df_cleaned = df_cleaned.merge(projects_by_city, on='projectid', how='left')
    df_cleaned = df_cleaned.merge(projects_by_state, on='projectid', how='left')
    
    # Save the updated dataset with new features
    cleaned_output_path = "../outputs/selected_dataset_with_new_features.csv"
    df_cleaned.to_csv(cleaned_output_path, index=False)
    
    print(f"Feature engineering complete. Updated dataset saved to {cleaned_output_path}.")