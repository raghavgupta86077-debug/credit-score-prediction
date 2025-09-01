# Credit Score Prediction Project

## Overview
This project implements a machine learning pipeline for predicting credit scores using a dataset from Kaggle. The notebook (`credit-score-prediction.ipynb`) performs data loading, cleaning, preprocessing, model training with a Random Forest Classifier, and prediction on test data. The goal is to classify credit scores into categories (e.g., Good, Standard, Poor) based on features like age, income, credit history, and payment behavior.

The dataset is a simplified version focusing on key features such as income, loan amount, term, credit history, and default status. The pipeline handles data issues like invalid values, missing data, and categorical encoding.

Key outcomes:
- Validation accuracy: ~78% on a hold-out set.
- Model saved as `credit_score_pipeline.pkl` for reuse.

## Dataset
- **Source**: Kaggle dataset - [Credit Score Classification by parisrohan](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- **Files Used**:
  - `train.csv`: 100,000 rows, 28 columns (includes target `Credit_Score`).
  - `test.csv`: 50,000 rows, 27 columns.
- **Key Features**:
  - Numeric: Age, Annual_Income, Monthly_Inhand_Salary, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment, Changed_Credit_Limit, Num_Credit_Inquiries, Outstanding_Debt, Credit_Utilization_Ratio, Credit_History_Age, Total_EMI_per_month, Amount_invested_monthly, Monthly_Balance.
  - Categorical: Occupation, Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour.
- **Target**: `Credit_Score` (encoded as 0: Good, 1: Standard, 2: Poor).
- **Data Issues Handled**: Invalid strings (e.g., '_'), negative/outlier ages, conversion of credit history to months.

The notebook downloads the dataset using `kagglehub` and assumes access to Kaggle APIs.

## Dependencies
- Python 3.12+
- Libraries (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - joblib
  - kagglehub (for dataset download)

Create a `requirements.txt` file with:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
kagglehub
```

## Installation
1. Clone the repository:
   ```
   git clone <repo-url>
   cd credit-score-prediction
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure Kaggle API credentials are set up for dataset download (see [Kaggle API docs](https://www.kaggle.com/docs/api)).

## Usage
1. **Run the Notebook**:
   - Open `credit-score-prediction.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute cells sequentially.
   - The notebook will:
     - Download and load the dataset.
     - Clean and preprocess data.
     - Train a Random Forest model via a scikit-learn pipeline.
     - Validate on a train-test split (accuracy printed).
     - Predict on the test set and print sample predictions.
     - Save the trained pipeline as `credit_score_pipeline.pkl`.

2. **Load and Use the Saved Model**:
   ```python
   import joblib
   import pandas as pd

   # Load the pipeline
   model = joblib.load('credit_score_pipeline.pkl')

   # Example: Predict on new data (replace with your DataFrame)
   new_data = pd.DataFrame({...})  # Must match training features
   predictions_encoded = model.predict(new_data)
   # Decode if needed (labels: 0=Good, 1=Standard, 2=Poor)
   ```

3. **Customization**:
   - Modify the pipeline in the notebook for different models (e.g., replace RandomForestClassifier).
   - Hyperparameter tuning can be added using GridSearchCV.

## Project Structure
- `credit-score-prediction.ipynb`: Main Jupyter notebook with all code.
- `credit_score_pipeline.pkl`: Saved model (generated after running the notebook).
- `README.md`: This file.
- (Optional) Dataset files: Downloaded automatically to `~/.cache/kagglehub/` or specified path.

## Model Details
- **Preprocessing**:
  - Custom `DataCleaner`: Handles string cleaning, invalid values, and conversions (e.g., Credit_History_Age to months).
  - Drops irrelevant columns: ID, Customer_ID, Month, Name, SSN, Type_of_Loan.
  - Numeric: Mean imputation + Standard scaling.
  - Categorical: Most-frequent imputation + One-hot encoding.
- **Classifier**: RandomForestClassifier (default params, random_state=42).
- **Evaluation**: Train-test split (80/20), accuracy metric.
- **Known Warnings**: Sklearn may warn about missing values in 'Credit_History_Age' during imputation—handled in the pipeline.

## Results
- Validation Accuracy: 0.7811 (on 20% hold-out set).
- Sample Test Predictions: Printed in the notebook (e.g., [2, 2, 2, 0, 2] corresponding to Poor, Poor, Poor, Good, Poor).

## Limitations
- No hyperparameter tuning or cross-validation in the base notebook.
- Assumes dataset is downloaded successfully; handle large files carefully.
- Test predictions are generated but not saved/submitted (add code if needed for Kaggle submission).




