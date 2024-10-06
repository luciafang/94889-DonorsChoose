# DonorsChoose Funding Success Prediction

This project contains a machine learning pipeline for predicting the funding success of DonorsChoose projects, with a particular focus on the impact of poverty levels.

## Project Overview

DonorsChoose is a platform where teachers can request resources for their classrooms. This project aims to predict whether a project will be successfully funded based on various features, including poverty levels of the school districts.

**Note:** The machine learning components of this pipeline are not yet fully implemented. The current state focuses on data preprocessing and feature engineering.

## Project Structure

- `run_pipeline.py`: Main script to run the entire pipeline
- `config.json`: Configuration file for dataset paths and feature selection
- `data/`: Directory containing all raw datasets
- `preprocessing/`: Directory containing preprocessing scripts
  - `feature_selection.py`: Script for merging datasets and selecting features
  - `data_cleaning.py`: Script for cleaning and preprocessing the data
- `outputs/`: Directory where processed datasets are saved
- `models/`: Directory for future machine learning models (to be implemented)
- `features/`: Directory for feature engineering scripts (to be implemented)

## Getting Started

1. Clone this repository to your local machine.
2. Ensure you have Python installed (preferably Python 3.7+).
3. Install the required dependencies:
   ```
   pip install pandas numpy
   ```
4. Place your raw DonorsChoose datasets in the `data/` directory (Make sure it match the name in `config.json`).
5. Review and update the `config.json` file if necessary.

## Running the Pipeline

You can run the entire pipeline using the `run_pipeline.py` script:

```
python run_pipeline.py
```

This will execute the following steps:
1. Merge datasets and select features
2. Clean and preprocess the data

Alternatively, you can run the scripts individually in the following order:

1. `python preprocessing/feature_selection.py`
2. `python preprocessing/data_cleaning.py`

## Configuration

The `config.json` file contains important settings for the pipeline:

- `raw_datasets`: Paths to the input CSV files (donations, essays, projects, resources, outcomes)
- `dataset`: Path for the output cleaned dataset
- `features_to_use`: List of features to select from the merged dataset, including poverty_level
- `one_hot_encode_features`: List of categorical features to one-hot encode

## Output

The pipeline generates two main output files in the `outputs/` directory:

1. `selected_dataset.csv`: Dataset with selected features after merging
2. `cleaned_dataset.csv`: Final cleaned and preprocessed dataset
