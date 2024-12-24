import pandas as pd
import numpy as np
import random

np.random.seed(12)
random.seed(12)

numerical_columns = ['id',
        'Age',
        'Annual Income',
        'Number of Dependents',
        'Health Score',
        'Previous Claims',
        'Vehicle Age',
        'Credit Score',
        'Insurance Duration'
 ]
def num_data_cleaner(data):
    data = data[numerical_columns]
    """
    Cleans the given dataset by filling missing values according to specific rules:
    - 'Age' is filled with the median.
    - 'Annual Income' is filled with the median.
    - 'Number of Dependents' is filled with 0 (assuming unknown implies none).
    - 'Health Score' is filled using linear interpolation.
    - 'Previous Claims' is filled randomly with values between 0 and 3.
    - 'Vehicle Age' is filled with the median.
    - 'Credit Score' is filled with the minimum value of the column.

    Args:
        data (pd.DataFrame): The dataset to clean.

    Returns:
        pd.DataFrame: The cleaned dataset with missing values filled.
    """
    
    np.random.seed(12)
    
    # Fill 'Age' with median
    if 'Age' in data.columns:
        data['Age'] = data['Age'].fillna(data['Age'].median())

    # Fill 'Annual Income' with median
    if 'Annual Income' in data.columns:
        data['Annual Income'] = data['Annual Income'].fillna(data['Annual Income'].median())

    # Fill 'Number of Dependents' with 0
    if 'Number of Dependents' in data.columns:
        data['Number of Dependents'] = data['Number of Dependents'].fillna(0)

    # Fill 'Health Score' with linear interpolation
    if 'Health Score' in data.columns:
        data['Health Score'] = data['Health Score'].interpolate(method='linear', limit_direction='forward')

    # Fill 'Previous Claims' with random integers between 0 and 3
    if 'Previous Claims' in data.columns:
        missing_claims = data['Previous Claims'].isna()
        data.loc[missing_claims, 'Previous Claims'] = np.random.randint(0, 4, size=missing_claims.sum())

    # Fill 'Vehicle Age' with median
    if 'Vehicle Age' in data.columns:
        data['Vehicle Age'] = data['Vehicle Age'].fillna(data['Vehicle Age'].median())

    # Fill 'Credit Score' with the minimum value
    if 'Credit Score' in data.columns:
        min_credit_score = data['Credit Score'].min()
        data['Credit Score'] = data['Credit Score'].fillna(min_credit_score)

    return data

def cat_data_cleaner(data):
    """
    Fills missing values for categorical columns based on specific logic:
    - 'Marital Status': Replaced with mode.
    - 'Occupation': Replaced with 'Unknown'.
    - 'Customer Feedback': Replaced with 'Average'.

    Args:
        data (pd.DataFrame): The dataset to process.

    Returns:
        pd.DataFrame: The dataset with missing values replaced.
    """
    # Fill 'Marital Status' with mode
    if 'Marital Status' in data.columns:
        marital_mode = data['Marital Status'].mode()[0]
        data['Marital Status'] = data['Marital Status'].fillna(marital_mode)
    
    # Fill 'Occupation' with 'Unknown'
    if 'Occupation' in data.columns:
        data['Occupation'] = data['Occupation'].fillna('Unknown')
    
    # Fill 'Customer Feedback' with 'Average'
    if 'Customer Feedback' in data.columns:
        data['Customer Feedback'] = data['Customer Feedback'].fillna('Average')
    
    return data
