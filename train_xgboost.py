import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# CONFIG
TRAIN_PATH = "sesame-jci-stroke-prediction/train.csv"
TEST_PATH = "sesame-jci-stroke-prediction/test.csv"
SUBMISSION_DIR = "submissions"
MODEL_DIR = "models"

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("1. Loading Data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    test_ids = test_df['id']
    
    # Combine for processing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['stroke'] = -1
    
    combined = pd.concat([train_df, test_df], axis=0)
    
    print("2. Preprocessing & Feature Engineering...")
    
    # --- Missing Values ---
    # BMI: Use median
    combined['bmi'] = combined['bmi'].fillna(combined['bmi'].median())
    
    # --- Feature Engineering ---
    # 1. Age Groups (Risk increases with age)
    combined['age_group'] = pd.cut(combined['age'], bins=[0, 18, 35, 50, 65, 100], labels=False)
    
    # 2. Glucose Risk (Diabetic levels)
    combined['glucose_risk'] = pd.cut(combined['avg_glucose_level'], bins=[0, 100, 140, 300], labels=False)
    
    # 3. BMI Categories (Underweight, Normal, Overweight, Obese)
    combined['bmi_cat'] = pd.cut(combined['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=False)
    
    # 4. Interaction: Age * Glucose (High glucose is worse for older people)
    combined['age_x_glucose'] = combined['age'] * combined['avg_glucose_level']
    
    # 5. Interaction: Age * BMI
    combined['age_x_bmi'] = combined['age'] * combined['bmi']
    
    # 6. High Risk Flag (Simple heuristic)
    combined['high_risk_flag'] = (
        (combined['age'] > 60) & 
        (combined['avg_glucose_level'] > 200) & 
        (combined['hypertension'] == 1)
    ).astype(int)

    # --- Encoding ---
    # One-Hot Encoding for categorical
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    
    # Drop ID
    combined = combined.drop('id', axis=1)
    
    # --- Split Back ---
    X_train = combined[combined['is_train'] == 1].drop(['stroke', 'is_train'], axis=1)
    y_train = combined[combined['is_train'] == 1]['stroke']
    X_test = combined[combined['is_train'] == 0].drop(['stroke', 'is_train'], axis=1)
    
    print(f"   Training Data Shape: {X_train.shape}")
    
    # --- Model Training (XGBoost) ---
    print("3. Training XGBoost Model...")
    
    # Calculate scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_weight = neg / pos
    print(f"   Class Imbalance Ratio: 1:{scale_weight:.1f} (Using scale_pos_weight)")

    # XGBoost Classifier
    # Tuned hyperparameters for imbalanced tabular data
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        min_child_weight=3,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        n_jobs=-1,
        scale_pos_weight=scale_weight, # Critical for imbalance
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=50
    )
    
    # Split for validation (to monitor early stopping)
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    # --- Evaluation ---
    print("4. Evaluating...")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"   Validation AUC: {val_auc:.5f}")
    
    # Find Best Threshold for F1
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (val_probs >= thresh).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"   Best Threshold (F1): {best_thresh:.2f}")
    print(f"   Best Validation F1: {best_f1:.5f}")
    
    # --- Predictions ---
    print("5. Generating Submissions...")
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Submission 1: Probabilities (Best for AUC)
    sub_prob = pd.DataFrame({'id': test_ids, 'stroke': test_probs})
    path_prob = f"{SUBMISSION_DIR}/submission_xgboost_prob.csv"
    sub_prob.to_csv(path_prob, index=False)
    print(f"   Saved PROBABILITY submission to: {path_prob} (Likely best for Kaggle)")
    
    # Submission 2: Labels (Best for Accuracy/F1)
    test_labels = (test_probs >= best_thresh).astype(int)
    sub_label = pd.DataFrame({'id': test_ids, 'stroke': test_labels})
    path_label = f"{SUBMISSION_DIR}/submission_xgboost_labels.csv"
    sub_label.to_csv(path_label, index=False)
    print(f"   Saved LABEL submission to: {path_label} (Use if metric is F1/Accuracy)")
    
    # Save Model
    joblib.dump(model, f"{MODEL_DIR}/model_xgboost.pkl")

if __name__ == "__main__":
    main()
