import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

# CONFIG
TRAIN_PATH = "sesame-jci-stroke-prediction/train.csv"
TEST_PATH = "sesame-jci-stroke-prediction/test.csv"
SUBMISSION_DIR = "submissions"
MODEL_DIR = "models"

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_data(df):
    # 1. Handling Missing Values (KNN is better than Median)
    # We need to encode first for KNN
    
    # Feature Engineering
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100], labels=False)
    df['glucose_risk'] = pd.cut(df['avg_glucose_level'], bins=[0, 100, 140, 300], labels=False)
    df['bmi_cat'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=False)
    
    # Interactions
    df['age_x_glucose'] = df['age'] * df['avg_glucose_level']
    df['age_x_bmi'] = df['age'] * df['bmi'] # Note: BMI might be NaN here, handled later
    
    # High Risk Flag
    df['high_risk_flag'] = (
        (df['age'] > 60) & 
        (df['avg_glucose_level'] > 200) & 
        (df['hypertension'] == 1)
    ).astype(int)
    
    return df

def main():
    print("1. Loading Data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    test_ids = test_df['id']
    
    # Combine
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['stroke'] = -1
    combined = pd.concat([train_df, test_df], axis=0)
    
    print("2. Advanced Preprocessing...")
    # Feature Engineering
    combined = preprocess_data(combined)
    
    # One-Hot Encoding
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    combined = combined.drop('id', axis=1)
    
    # Imputation (KNN)
    # We impute AFTER encoding so KNN can use all features
    print("   Running KNN Imputation (this might take a moment)...")
    imputer = KNNImputer(n_neighbors=5)
    cols = combined.columns
    combined_imputed = pd.DataFrame(imputer.fit_transform(combined), columns=cols)
    
    # Split back
    X_train = combined_imputed[combined_imputed['is_train'] == 1].drop(['stroke', 'is_train'], axis=1)
    y_train = combined_imputed[combined_imputed['is_train'] == 1]['stroke']
    X_test = combined_imputed[combined_imputed['is_train'] == 0].drop(['stroke', 'is_train'], axis=1)
    
    print(f"   Training Shape: {X_train.shape}")

    # --- Models ---
    print("3. Training Ensemble Models...")
    
    # Model 1: XGBoost (The one that got 0.80)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_weight = neg / pos
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        min_child_weight=3,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        n_jobs=-1,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='auc'
    )
    
    # Model 2: Random Forest with SMOTE
    # RF needs SMOTE to handle imbalance well
    rf_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    # Training XGBoost
    print("   Training XGBoost...")
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Training Random Forest
    print("   Training Random Forest (with SMOTE)...")
    rf_pipeline.fit(X_train, y_train)
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    
    # --- Ensemble (Weighted Average) ---
    print("4. Creating Ensemble...")
    # XGBoost is usually stronger, so we give it more weight (e.g., 0.7)
    # But since RF is decent, a 0.6/0.4 split is often safe.
    # Let's try 0.6 XGB + 0.4 RF
    ensemble_probs = (0.6 * xgb_probs) + (0.4 * rf_probs)
    
    # --- Submission ---
    print("5. Saving Submission...")
    sub = pd.DataFrame({'id': test_ids, 'stroke': ensemble_probs})
    path = f"{SUBMISSION_DIR}/submission_ensemble_v1.csv"
    sub.to_csv(path, index=False)
    
    print(f"   Saved to {path}")
    print("   Done! Upload this file to try to beat 0.80.")

if __name__ == "__main__":
    main()
