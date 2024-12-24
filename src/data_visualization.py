import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



def visualize_data_distribution(data):
    """
    Visualizes the distribution of the data in the columns (excluding 'id').

    For numerical columns:
    - Plots histograms to show data distribution.
    - Plots boxplots to check for outliers.

    For categorical columns:
    - Plots bar plots.

    Args:
        data (pd.DataFrame): The dataset to visualize.
    """
    # Exclude 'id' column
    data = data.drop(columns=['id'], errors='ignore')
    
    # Identify numerical and categorical columns
    numerical_columns = data.select_dtypes(include=['number']).columns

    # Plot numerical data
    for column in numerical_columns:
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(data[column], kde=True, bins=30, color='blue')
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[column], color='green')
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)
        
        plt.tight_layout()
        plt.show()
    
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix")
    plt.show()



def visualize_categorical_data(data):
    """
    Visualizes the distribution of categorical columns in the dataset using bar plots.

    Args:
        data (pd.DataFrame): The dataset to visualize.

    Returns:
        None: Displays the bar plots for categorical columns.
    """
    # Identify categorical columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    
    for column in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=column, palette='viridis', order=data[column].value_counts().index)
        plt.title(f"Distribution of {column}")
        plt.xlabel("Count")
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()
