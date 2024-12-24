import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
import joblib  
from data_cleaner import num_data_cleaner, cat_data_cleaner

def extract_date_info(data, date_column="Policy Start Date"):
    """
    Extracts detailed information from a datetime column and adds it as new columns.

    Args:
        data (pd.DataFrame): The dataset containing the datetime column.
        date_column (str): The name of the datetime column to process.

    Returns:
        pd.DataFrame: The dataset with additional columns for extracted date information.
    """
    # Ensure the column is in datetime format
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Extract Year, Month, Day
    data['Year'] = data[date_column].dt.year
    data['Month'] = data[date_column].dt.month
    data['Day'] = data[date_column].dt.day
    
    
    # Extract Day of the Week (Monday=0, Sunday=6)
    data['Day of Week'] = data[date_column].dt.dayofweek
    
    # Extract whether the day is a weekend (Saturday=5, Sunday=6)
    data['Is Weekend'] = data['Day of Week'].apply(lambda x: 1 if x in [5, 6] else 0)
    
    data = data.drop(columns=[date_column])
    
    return data



def preprocess_data(data):
    """
    Preprocesses the combined dataset:
    - Drops the 'id' column.
    - Leaves numerical features unchanged.
    - Encodes categorical and binary features.
    
    Args:
        data (pd.DataFrame): The combined dataset to preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    # Drop the 'id' column if it exists
    data = data.drop(columns=['id'], errors='ignore')
    
    # Separate numerical and categorical columns
    numerical_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns

    # Encode categorical features
    label_encoders = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        data[column] = encoder.fit_transform(data[column])
        label_encoders[column] = encoder
    
    return data


# Custom Transformers
class NumericalCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return num_data_cleaner(X)  # Reusing the earlier defined function


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return cat_data_cleaner(X)  # Reusing the earlier defined function


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.columns:
            X = extract_date_info(X, col)  # Reusing the earlier defined function
        return X


# Define the preprocessing pipeline
def build_pipeline(raw_data):
    # Define the transformers for each type of data
    num_transformer = Pipeline(steps=[
        ('cleaner', NumericalCleaner())
    ])

    cat_transformer = Pipeline(steps=[
        ('cleaner', CategoricalCleaner()),
        ('encoder', preprocess_data())
    ])

    date_transformer = Pipeline(steps=[
        ('extractor', DateFeatureExtractor())
    ])

    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, list(raw_data.select_dtypes(include=['number']).columns)),
            ('cat', cat_transformer, list(raw_data.select_dtypes(include=['object', 'category']).columns)),
            ('date', date_transformer, list(raw_data.select_dtypes(include=['datetime64', 'object']).filter(regex="date|Date", axis=1).columns))
        ]
    )

    # Create the final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('preprocess_data', FunctionTransformer(preprocess_data))  # Reusing preprocess_data function
    ])

    return pipeline