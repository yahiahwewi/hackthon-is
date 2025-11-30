"""
Module: Data Loader
===================
Purpose:
    This module handles the ingestion and initial cleaning of the raw medical dataset.
    We use Pandas to read the CSV file and perform basic filtering.

Objective:
    - Load the 'sesame-jci-stroke-prediction' dataset (train.csv).
    - Remove non-predictive identifiers (e.g., 'id').
    - Filter out data inconsistencies (e.g., 'Other' gender which has only 1 record).

Goal:
    To provide a clean, raw DataFrame that is ready for feature engineering and 
    preprocessing, ensuring no garbage data enters the pipeline.
"""
import pandas as pd

def load_data(path):
    """
    Load data from CSV.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File {path} not found.")

def clean_data(df):
    """
    Perform initial cleaning and report data quality issues.
    """
    print("\n=== 📊 Data Quality & Cleaning Report ===")
    initial_rows = len(df)
    print(f"Initial Row Count: {initial_rows}")

    # 1. Check for ID column
    if 'id' in df.columns:
        print(f"-> 🗑️  Dropping 'id' column: It is a non-predictive unique identifier.")
        df = df.drop('id', axis=1)

    # 2. Check for 'Other' Gender
    other_gender_count = df[df['gender'] == 'Other'].shape[0]
    if other_gender_count > 0:
        print(f"-> ⚠️  Found {other_gender_count} record(s) with Gender='Other'.")
        print("    Action: Dropping these records to maintain binary gender consistency for the model.")
        df = df[df['gender'] != 'Other']
    else:
        print("-> ✅ No 'Other' gender records found.")

    # 3. Check for Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"-> ⚠️  Found {duplicates} duplicate rows.")
        print("    Action: Dropping duplicates to prevent data leakage.")
        df = df.drop_duplicates()
    else:
        print("-> ✅ No duplicate rows found.")

    # 4. Report Missing Values (to be handled later)
    missing_values = df.isnull().sum()
    missing_bmi = missing_values.get('bmi', 0)
    if missing_bmi > 0:
        print(f"-> ℹ️  Found {missing_bmi} missing values in 'bmi'.")
        print("    Action: Will be handled by KNN Imputation in the preprocessing pipeline.")

    final_rows = len(df)
    dropped_rows = initial_rows - final_rows
    print(f"Final Row Count: {final_rows} (Total Dropped: {dropped_rows})")
    print("=========================================\n")
    
    return df
