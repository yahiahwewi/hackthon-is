"""
Train models with different preprocessing methods and generate submission files.

This script:
1. Trains a separate model for each preprocessing variant (KNN, Median)
2. Evaluates on validation set
3. Generates predictions on the test set
4. Creates submission files (id, stroke) for each method
5. Produces a comparison report
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report

# Local imports
from src.data.loader import load_data, clean_data
from src.features.preprocessing_knn import get_preprocessor as get_knn_preprocessor
from src.features.preprocessing_median import get_preprocessor as get_median_preprocessor

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
TRAIN_PATH = "sesame-jci-stroke-prediction/train.csv"
TEST_PATH = "sesame-jci-stroke-prediction/test.csv"
OUTPUT_DIR = "models"
SUBMISSION_DIR = "submissions"
PLOT_DIR = "plots"
REPORT_FILE = "PREPROCESSING_COMPARISON.md"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Load & split training data
# ---------------------------------------------------------------------
print("Loading training data...")
raw_train = load_data(TRAIN_PATH)
clean_train = clean_data(raw_train)
X = clean_train.drop("stroke", axis=1)
y = clean_train["stroke"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load test data
print("Loading test data...")
raw_test = pd.read_csv(TEST_PATH)
test_ids = raw_test['id'].copy()  # Save IDs for submission
# Drop id from test features
X_test = raw_test.drop('id', axis=1)

# Handle 'Other' gender in test if it exists
if 'gender' in X_test.columns:
    # Replace 'Other' with most common gender to avoid issues
    if (X_test['gender'] == 'Other').any():
        most_common_gender = X_train['gender'].mode()[0]
        X_test.loc[X_test['gender'] == 'Other', 'gender'] = most_common_gender

# ---------------------------------------------------------------------
# Define preprocessing variants
# ---------------------------------------------------------------------
preprocessing_methods = {
    "knn": get_knn_preprocessor(),
    "median": get_median_preprocessor()
}

results = []

for method_name, preprocessor in preprocessing_methods.items():
    print(f"\n{'='*60}")
    print(f"Training with {method_name.upper()} preprocessing")
    print(f"{'='*60}")
    
    # Build pipeline
    pipeline = ImbPipeline([
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_depth=15
        )),
    ])

    # Fit on training split
    print("Fitting model...")
    pipeline.fit(X_train, y_train)

    # ---- Evaluation on TRAIN set ----
    print("Evaluating on train set...")
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_pred = (train_proba >= 0.5).astype(int)
    train_auc = roc_auc_score(y_train, train_proba)
    train_f1 = f1_score(y_train, train_pred)

    # ---- Evaluation on VALIDATION set ----
    print("Evaluating on validation set...")
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    val_auc = roc_auc_score(y_val, val_proba)
    val_f1 = f1_score(y_val, val_pred)

    print(f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
    print(f"Train F1:  {train_f1:.4f} | Val F1:  {val_f1:.4f}")

    # ---- Generate predictions on TEST set ----
    print("Generating test predictions...")
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    # Create submission file
    submission = pd.DataFrame({
        'id': test_ids,
        'stroke': test_pred
    })
    submission_path = os.path.join(SUBMISSION_DIR, f"submission_{method_name}.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to: {submission_path}")

    # Save model artifact
    artifact = {
        "pipeline": pipeline,
        "base_threshold": 0.5,
        "method": method_name
    }
    model_path = os.path.join(OUTPUT_DIR, f"model_{method_name}.pkl")
    joblib.dump(artifact, model_path)
    print(f"Saved model to: {model_path}")

    # Record results
    results.append({
        "method": method_name,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "overfit_auc": train_auc - val_auc,
        "overfit_f1": train_f1 - val_f1
    })

# ---------------------------------------------------------------------
# Create comparison visualizations
# ---------------------------------------------------------------------
print("\nGenerating comparison plots...")
df_results = pd.DataFrame(results)

# Plot 1: AUC comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AUC plot
x = range(len(df_results))
width = 0.35
axes[0].bar([i - width/2 for i in x], df_results["train_auc"], width, 
            label="Train AUC", color="#38bdf8", alpha=0.8)
axes[0].bar([i + width/2 for i in x], df_results["val_auc"], width, 
            label="Val AUC", color="#818cf8", alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(df_results["method"].str.upper())
axes[0].set_ylabel("AUC Score")
axes[0].set_title("AUC: Train vs Validation")
axes[0].set_ylim(0.8, 1.0)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# F1 plot
axes[1].bar([i - width/2 for i in x], df_results["train_f1"], width, 
            label="Train F1", color="#22c55e", alpha=0.8)
axes[1].bar([i + width/2 for i in x], df_results["val_f1"], width, 
            label="Val F1", color="#ef4444", alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(df_results["method"].str.upper())
axes[1].set_ylabel("F1 Score")
axes[1].set_title("F1: Train vs Validation")
axes[1].set_ylim(0, 1.0)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, "preprocessing_comparison.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------
# Write markdown report
# ---------------------------------------------------------------------
print("Writing comparison report...")
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("# 📊 Preprocessing Methods Comparison\n\n")
    f.write("This report compares different preprocessing strategies for handling missing BMI values.\n\n")
    
    f.write("## 🎯 Objective\n")
    f.write("Train separate models using different imputation methods and evaluate:\n")
    f.write("- **Performance** (AUC-ROC, F1-Score)\n")
    f.write("- **Overfitting** (Train vs Validation gap)\n")
    f.write("- **Submission quality** (predictions on test set)\n\n")
    
    f.write("## 📈 Results Summary\n\n")
    f.write("| Method | Train AUC | Val AUC | Train F1 | Val F1 | Overfit Δ AUC | Overfit Δ F1 |\n")
    f.write("|--------|-----------|---------|----------|--------|---------------|-------------|\n")
    
    for r in results:
        f.write(f"| **{r['method'].upper()}** | "
                f"{r['train_auc']:.4f} | {r['val_auc']:.4f} | "
                f"{r['train_f1']:.4f} | {r['val_f1']:.4f} | "
                f"{r['overfit_auc']:.4f} | {r['overfit_f1']:.4f} |\n")
    
    f.write("\n## 📊 Visual Comparison\n\n")
    f.write(f"![Preprocessing Comparison]({plot_path})\n\n")
    
    f.write("## 🔍 Analysis\n\n")
    f.write("### Overfitting Detection\n")
    f.write("- **Δ AUC/F1 < 0.02**: Minimal overfitting ✅\n")
    f.write("- **Δ AUC/F1 0.02-0.05**: Moderate overfitting ⚠️\n")
    f.write("- **Δ AUC/F1 > 0.05**: Significant overfitting ❌\n\n")
    
    # Find best method
    best_method = df_results.loc[df_results['val_auc'].idxmax(), 'method']
    best_auc = df_results.loc[df_results['val_auc'].idxmax(), 'val_auc']
    
    f.write(f"### 🏆 Best Method: **{best_method.upper()}**\n")
    f.write(f"- Validation AUC: **{best_auc:.4f}**\n")
    f.write(f"- Submission file: `submissions/submission_{best_method}.csv`\n\n")
    
    f.write("## 📁 Generated Files\n\n")
    f.write("### Models\n")
    for method in preprocessing_methods.keys():
        f.write(f"- `models/model_{method}.pkl`\n")
    
    f.write("\n### Submissions\n")
    for method in preprocessing_methods.keys():
        f.write(f"- `submissions/submission_{method}.csv` (format: id, stroke)\n")
    
    f.write("\n## 💡 Recommendations\n\n")
    f.write(f"1. **For Kaggle submission**: Use `submission_{best_method}.csv`\n")
    f.write("2. **For production API**: Load `model_{}.pkl` in `main.py`\n".format(best_method))
    f.write("3. **For further improvement**: Consider ensemble methods or hyperparameter tuning\n")

print(f"\n{'='*60}")
print("✅ Benchmark complete!")
print(f"{'='*60}")
print(f"Report: {REPORT_FILE}")
print(f"Models: {OUTPUT_DIR}/")
print(f"Submissions: {SUBMISSION_DIR}/")
print(f"Plot: {plot_path}")
