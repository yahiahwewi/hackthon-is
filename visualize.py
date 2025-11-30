"""
Module: Visualization
=====================
Purpose:
    This script generates exploratory data analysis (EDA) plots to visualize the 
    population demographics, risk factor distributions, and correlations.

Objective:
    - Visualize the Class Imbalance (Stroke vs Healthy).
    - Analyze distributions of key features (Age, Glucose, BMI).
    - Explore relationships between risk factors and Stroke.

Goal:
    To provide visual insights that explain the data characteristics and justify 
    modeling decisions (like using SMOTE for imbalance).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data.loader import load_data, clean_data

# CONFIG
DATA_PATH = 'sesame-jci-stroke-prediction/train.csv'
OUTPUT_DIR = 'plots'

def save_plot(fig, name):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved plot: {path}")
    plt.close(fig)

def main():
    print("Loading Data for Visualization...")
    try:
        df = load_data(DATA_PATH)
        df = clean_data(df)
    except Exception as e:
        print(e)
        return

    # Set Style
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Target Distribution (Pie Chart)
    print("Generating Target Distribution Plot...")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    stroke_counts = df['stroke'].value_counts()
    ax1.pie(stroke_counts, labels=['Healthy', 'Stroke'], autopct='%1.1f%%', 
            colors=['#66b3ff', '#ff9999'], startangle=90, explode=(0, 0.1))
    ax1.set_title('Class Distribution: Stroke vs Healthy')
    save_plot(fig1, '01_target_distribution.png')

    # 2. Age Distribution by Stroke Status (KDE)
    print("Generating Age Distribution Plot...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x='age', hue='stroke', fill=True, common_norm=False, palette=['blue', 'red'], alpha=0.5, ax=ax2)
    ax2.set_title('Age Distribution by Stroke Status')
    ax2.set_xlabel('Age')
    save_plot(fig2, '02_age_distribution.png')

    # 3. Categorical Features Count (Gender, Work, Residence, Smoking)
    print("Generating Categorical Counts...")
    fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.countplot(data=df, x='gender', hue='stroke', ax=axes[0, 0])
    sns.countplot(data=df, x='work_type', hue='stroke', ax=axes[0, 1])
    sns.countplot(data=df, x='Residence_type', hue='stroke', ax=axes[1, 0])
    sns.countplot(data=df, x='smoking_status', hue='stroke', ax=axes[1, 1])
    
    axes[0, 0].set_title('Gender vs Stroke')
    axes[0, 1].set_title('Work Type vs Stroke')
    axes[1, 0].set_title('Residence Type vs Stroke')
    axes[1, 1].set_title('Smoking Status vs Stroke')
    plt.tight_layout()
    save_plot(fig3, '03_categorical_features.png')

    # 4. Glucose vs BMI Scatter
    print("Generating Glucose vs BMI Scatter...")
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='avg_glucose_level', y='bmi', hue='stroke', style='stroke', palette=['blue', 'red'], alpha=0.7, ax=ax4)
    ax4.set_title('Glucose vs BMI (Colored by Stroke)')
    save_plot(fig4, '04_glucose_vs_bmi.png')

    # 5. Correlation Matrix (Numerical)
    print("Generating Correlation Matrix...")
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
    ax5.set_title('Correlation Matrix (Numerical Features)')
    save_plot(fig5, '05_correlation_matrix.png')

    print("\n✅ All visualizations generated in 'plots/' directory.")

if __name__ == "__main__":
    main()
