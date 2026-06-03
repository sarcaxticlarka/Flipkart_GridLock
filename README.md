# Flipkart GridLock: Traffic Demand Prediction

Welcome to the Flipkart GridLock project. This repository contains the data science and machine learning pipeline for predicting traffic demand across various regions based on spatial, temporal, and environmental factors.

## Objective
The primary goal of this project is to accurately predict traffic `demand` using historical data. The dataset includes features such as road characteristics (RoadType, NumberofLanes), restrictions (LargeVehicles), environmental conditions (Temperature, Weather), and spatial-temporal data (geohash, day, timestamp).

## Project Structure

```text
Flipkart_GridLock/
├── dataset/                    # Directory for all data files
│   ├── train.csv               # Training data with demand target
│   ├── test.csv                # Test data for generating predictions
│   └── sample_submission.csv   # Expected format for the final output
├── notebooks/                  # Jupyter notebooks for analysis and modeling
│   ├── 01_train_eda.ipynb           # Exploratory Data Analysis of the training set
│   ├── 02_test_profile.ipynb        # Profiling and checking the test set
│   ├── 03_submission_format.ipynb   # Understanding the submission requirements
│   └── 04_modeling_submission.ipynb # Model training and prediction generation
├── outputs/                    # Stored predictions and outputs
│   ├── baseline_submission.csv
│   └── submission_hist_gradient_boosting.csv
├── pyproject.toml              # Python project configuration and dependencies
├── uv.lock                     # Lockfile for reproducible environments
└── README.md                   # Project documentation
```

## Setup Instructions

Follow these steps to set up the project locally on your machine.

### 1. Clone the repository
```bash
git clone  
cd Flipkart_GridLock
```

### 2. Create and activate a virtual environment
It is highly recommended to isolate the project dependencies using a virtual environment.
```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

### 3. Install dependencies
The project uses `pyproject.toml` to specify required packages. You can install them using the standard `pip` or using `uv`.
```bash
# Using standard pip
pip install .

# OR using uv (recommended for speed and reproducibility)
uv pip install -r pyproject.toml
```

### 4. Launch Jupyter Notebook
Once the dependencies are installed, you can launch the Jupyter environment to run the notebooks.
```bash
jupyter notebook
```
This will open a browser window. Navigate to the `notebooks/` directory and execute the notebooks in numerical order.

## Workflow

To understand the project or to reproduce the results, follow the notebooks in numerical order:

1. **Exploratory Data Analysis (EDA):** 
   Start with `notebooks/01_train_eda.ipynb` to understand data distributions, missing values, and the relationship between features and traffic demand.
2. **Test Set Profiling:** 
   Check `notebooks/02_test_profile.ipynb` to see how the test data compares against the training data and ensure there is no major feature drift.
3. **Format Validation:** 
   Use `notebooks/03_submission_format.ipynb` to familiarize yourself with the structure required for the final predictions.
4. **Modeling & Submission:** 
   Run `notebooks/04_modeling_submission.ipynb` to train machine learning models (like XGBoost or Gradient Boosting) on the dataset and generate the final `submission.csv` files in the `outputs/` directory.

## Data Overview

The dataset (`train.csv`) consists of the following key columns:
- **Spatial/Temporal:** `geohash`, `day`, `timestamp`
- **Road Properties:** `RoadType`, `NumberofLanes`, `Landmarks`
- **Traffic Rules:** `LargeVehicles`
- **Weather:** `Temperature`, `Weather`
- **Target:** `demand` (Float value representing traffic demand)

## Contributing
If you are a team member looking to contribute:
1. Create a new branch for your feature or experiment (`git checkout -b feature-name`).
2. Do not commit the `venv` directory or large datasets (ensure they remain in `.gitignore`).
3. Submit a Pull Request for review once your experiments are complete.
