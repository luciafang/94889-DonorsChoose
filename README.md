# DonorsChoose Funding Success Prediction

This project contains a machine learning pipeline for predicting the funding success of DonorsChoose projects, with a particular focus on the impact of poverty levels.

## Project Overview

DonorsChoose is a platform where teachers can request resources for their classrooms. This project aims to predict whether a project will be successfully funded based on various features, including poverty levels of the school districts.

## Project Structure

- `run_pipeline.py`: Main script to run the entire pipeline
- `config.json`: Configuration file for dataset paths and feature selection
- `data/`: Directory containing all raw datasets
- `preprocessing/`: Directory containing preprocessing scripts
  - `feature_selection.py`: Script for merging datasets and selecting features
  - `data_cleaning.py`: Script for cleaning and preprocessing the data
  - `data_segment_and_balance.py`: Script for segmenting and balancing the data
- `features/`: Directory for feature engineering scripts
  - `feature_engineering.py`: Script for feature engineering
- `split/`: Directory for data splitting scripts
  - `train_test_split.py`: Script for splitting data into training and testing sets
- `model/`: Directory for machine learning models, including training, validation, and selection scripts
  - `model_training.py`: Script for training the model
  - `model_evaluation.py`: Script for evaluating the model
  - `feature_importance.py`: Script for determining feature importance
  - `recommendation.py`: Script for generating recommendations
- `outputs/`: Directory where processed datasets are saved
- `figures/`: Directory for saving generated figures and plots
- `notebooks/`: Directory for Jupyter notebooks used for generating graphs and as a playground for experimentation

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
3. Merge datasets and select features
4. Perform feature engineering
5. Clean and preprocess the data
6. Split data into training and testing sets
7. Segment and balance the data
8. Train the machine learning model
9. Evaluate the model
10. Determine feature importance
11. Generate recommendations

Alternatively, you can run the scripts individually in the following order:

1. python preprocessing/feature_selection.py
2. python features/feature_engineering.py
3. python preprocessing/data_cleaning.py
4. python split/train_test_split.py
5. python preprocessing/data_segment_and_balance.py
6. python model/model_training.py
7. python model/model_evaluation.py
8. python model/feature_importance.py
9. python model/recommendation.py

## Configuration

The `config.json` file contains important settings for the pipeline:

- `raw_datasets`: Paths to the input CSV files (donations, essays, projects, resources, outcomes)
- `dataset`: Path for the output cleaned dataset
- `features_to_use`: List of features to select from the merged dataset
- `one_hot_encode_features`: List of categorical features to one-hot encode
- `models`: List of models to use
- `projects_imputation`: Methods for imputing missing values in the projects dataset
- `poverty_columns`: Mapping of poverty levels
- `split_by_poverty`: Whether to split data by poverty level
- `test_splits`: Configuration for test splits
- `poverty_level_replacements`: Replacements for poverty levels
- `quant_variables`: List of quantitative variables
- `stem_cols`: List of STEM-related columns

## Output

The pipeline generates several types of output files in the `outputs/` directory, including:

- CSV files: Containing datasets at various stages of processing (e.g., selected features, cleaned data, training and testing sets, model outputs)
- PKL files: Serialized model objects

Additionally, figures and plots generated during the analysis are saved in the `figures/` directory.
