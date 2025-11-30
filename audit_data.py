"""
Module: Data Quality Audit
==========================
Purpose:
    To analyze, visualize, and report data quality issues (defects) in the raw dataset.
    This ensures transparency about what data was modified or dropped.

Objective:
    - Identify specific defects: Missing Values, Inconsistent Categories, Duplicates.
    - Quantify the extent of these defects.
    - Visualize the impact on the dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CONFIG
DATA_PATH = 'sesame-jci-stroke-prediction/train.csv'
OUTPUT_DIR = 'plots/audit'
REPORT_FILE = 'DATA_AUDIT.md'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def audit_data():
    print("Starting Data Quality Audit...")
    ensure_dir(OUTPUT_DIR)
    
    # Load Raw Data
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File {DATA_PATH} not found.")
        return

    total_rows = len(df)
    defects = {
        'Missing BMI': 0,
        'Gender "Other"': 0,
        'Duplicates': 0,
        'Clean': 0
    }

    # 1. Identify Defects
    # Note: A row can have multiple defects, but for simplicity in "Clean vs Defective" split,
    # we check if a row has ANY defect.
    
    # Mask for defects
    mask_missing_bmi = df['bmi'].isnull()
    mask_gender_other = df['gender'] == 'Other'
    mask_duplicates = df.duplicated()

    # Counts
    defects['Missing BMI'] = mask_missing_bmi.sum()
    defects['Gender "Other"'] = mask_gender_other.sum()
    defects['Duplicates'] = mask_duplicates.sum()

    # Total Defective Rows (Union of defects)
    mask_any_defect = mask_missing_bmi | mask_gender_other | mask_duplicates
    total_defective = mask_any_defect.sum()
    defects['Clean'] = total_rows - total_defective

    print(f"Audit Complete. Found {total_defective} defective rows out of {total_rows}.")

    # ==========================================
    # 2. VISUALIZATION
    # ==========================================
    sns.set_theme(style="whitegrid")

    # Plot 1: Defect Types (Bar Chart)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    defect_counts = {k: v for k, v in defects.items() if k != 'Clean'}
    sns.barplot(x=list(defect_counts.keys()), y=list(defect_counts.values()), palette="viridis", ax=ax1)
    ax1.set_title('Count of Data Defects by Type')
    ax1.set_ylabel('Number of Rows')
    for i, v in enumerate(defect_counts.values()):
        ax1.text(i, v + 5, str(v), ha='center')
    
    fig1.savefig(os.path.join(OUTPUT_DIR, '01_defect_types.png'), dpi=300)
    plt.close(fig1)

    # Plot 2: Clean vs Defective (Pie Chart)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.pie([defects['Clean'], total_defective], labels=['Clean Data', 'Defective Data'], 
            autopct='%1.1f%%', colors=['#4ade80', '#f87171'], startangle=90, explode=(0, 0.1))
    ax2.set_title('Data Quality Overview: Clean vs Defective Rows')
    
    fig2.savefig(os.path.join(OUTPUT_DIR, '02_data_quality_pie.png'), dpi=300)
    plt.close(fig2)

    # ==========================================
    # 3. GENERATE REPORT (Markdown)
    # ==========================================
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# 🛡️ Data Quality Audit Report\n\n")
        f.write("This document details the data quality issues found in the raw dataset and the actions taken to resolve them.\n\n")
        
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Total Rows Analyzed:** {total_rows}\n")
        f.write(f"- **Clean Rows:** {defects['Clean']} ({defects['Clean']/total_rows:.1%})\n")
        f.write(f"- **Defective Rows:** {total_defective} ({total_defective/total_rows:.1%})\n\n")
        f.write("![Data Quality Pie](plots/audit/02_data_quality_pie.png)\n\n")

        f.write("## 2. Defect Breakdown\n")
        f.write("We identified the following specific issues:\n\n")
        f.write("| Defect Type | Count | Percentage | Impact | Action Taken |\n")
        f.write("|-------------|-------|------------|--------|--------------|\n")
        
        # Missing BMI
        pct_bmi = defects['Missing BMI'] / total_rows
        f.write(f"| **Missing BMI** | {defects['Missing BMI']} | {pct_bmi:.1%} | Critical Feature Missing | **Imputed** using KNN (k=5) |\n")
        
        # Gender Other
        pct_gender = defects['Gender "Other"'] / total_rows
        f.write(f"| **Gender 'Other'** | {defects['Gender \"Other\"']} | {pct_gender:.1%} | Inconsistent Category | **Dropped** (Sample size too small) |\n")
        
        # Duplicates
        pct_dup = defects['Duplicates'] / total_rows
        f.write(f"| **Duplicates** | {defects['Duplicates']} | {pct_dup:.1%} | Data Leakage Risk | **Dropped** |\n")
        
        f.write("\n![Defect Types](plots/audit/01_defect_types.png)\n\n")

        f.write("## 3. Resolution Strategy\n")
        f.write("### A. Missing BMI (Imputation)\n")
        f.write("Instead of dropping 4% of our data (which is significant), we used **KNN Imputation**. This algorithm finds the 5 most similar patients (based on Age, Glucose, etc.) and uses their average BMI to fill the gap. This preserves statistical integrity.\n\n")
        
        f.write("### B. Gender 'Other' (Dropping)\n")
        f.write("With only 1 occurrence, the 'Other' category provides insufficient statistical power for the model to learn anything meaningful. Dropping it prevents noise.\n")

    print(f"Report generated: {REPORT_FILE}")

if __name__ == "__main__":
    audit_data()
