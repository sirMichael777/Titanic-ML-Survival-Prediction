# Titanic Survival Prediction

This project uses machine learning to predict the survival of passengers on the Titanic. The final model is a Random Forest Classifier with hyperparameters tuned using GridSearchCV.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Predicting Test Data](#predicting-test-data)
- [Results](#results)
- [Contributing](#contributing)


## Project Overview

The goal of this project is to build a machine learning model to predict whether a passenger on the Titanic survived or not, based on various features such as age, gender, class, etc.

## Data

The dataset is from the famous Kaggle competition "Titanic: Machine Learning from Disaster". It includes two CSV files: `train.csv` and `test.csv`.

The data files should be placed in the `data` directory:

- `data/train.csv`
- `data/test.csv`

## Setup and Installation

1. Clone the repository:

2. Create a virtual environment:

    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:

        ```sh
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```sh
        source venv/bin/activate
        ```

4. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Preprocess the data and train the model by running the notebook `titanic_analysis.ipynb` located in the `src` directory.

2. Make predictions on the test data:
    - Ensure that the model and preprocessor are saved.
    - Run the `titanic_analysis.ipynb` notebook cells to make predictions.

## Model Training

The model training process involves the following steps:

1. **Data Exploration and Visualization:**
    - Load the training data.
    - Check for missing values and summary statistics.
    - Visualize the data using boxplots for numerical features.

2. **Data Preprocessing:**
    - Handle missing values:
        - Replace missing `Age` values with the median age.
        - Replace missing `Embarked` values with the most common value.
        - Drop the `Cabin` column due to many missing values.
    - Convert categorical data to numerical using one-hot encoding.
    - Drop columns not useful for prediction (`Name`, `Ticket`, `PassengerId`).

3. **Model Training:**
    - Split the data into training and validation sets.
    - Train multiple models (Logistic Regression, Decision Tree, Random Forest) and evaluate their performance.

## Hyperparameter Tuning

GridSearchCV is used for hyperparameter tuning. The parameter grid includes variations in:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `bootstrap`

The best parameters are selected based on cross-validation performance.

## Predicting Test Data

The final model is trained on the entire training dataset using the best hyperparameters. Predictions are then made on the test dataset.

Ensure the test data preprocessing steps match those applied to the training data:

- Handle missing values.
- Convert categorical features.
- Align test data columns with training data.

## Results

The predictions are saved in a CSV file named `results.csv` located in the `src` directory with the following columns:

- `PassengerId`
- `Survived`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
