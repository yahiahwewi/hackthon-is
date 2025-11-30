"""
Advanced Training with CatBoost and LightGBM
Goal: Push AUC to 0.85+ using state-of-the-art gradient boosting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# CONFIG
TRAIN_PATH = "sesame-jci-stroke-prediction/train.csv"
TEST_PATH = "sesame-jci-stroke-prediction/test.csv"
SUBMISSION_DIR = "submissions"
MODEL_DIR = "models"
RANDOM_STATE = 42

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def advanced_feature_engineering(df):
    """Create advanced features"""
    
    # Age-based risk groups
    df['age_risk_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 75, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    
    # Glucose categories
    df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins=[0, 100, 125, 200, 300], labels=[0, 1, 2, 3]).astype(int)
    
    # BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 40, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    
    # Interaction Features
    df['age_glucose_interaction'] = df['age'] * df['avg_glucose_level']
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['glucose_bmi_interaction'] = df['avg_glucose_level'] * df['bmi']
    
    # Polynomial features
    df['age_squared'] = df['age'] ** 2
    df['glucose_squared'] = df['avg_glucose_level'] ** 2
    df['bmi_squared'] = df['bmi'] ** 2
    
    # High-risk flags
    df['is_senior'] = (df['age'] > 65).astype(int)
    df['is_diabetic_range'] = (df['avg_glucose_level'] > 126).astype(int)
    df['is_obese'] = (df['bmi'] > 30).astype(int)
    df['is_hypertensive_obese'] = (df['hypertension'] * df['is_obese']).astype(int)
    
    # Cardiovascular risk score
    df['cardio_risk_score'] = (
        df['hypertension'] * 2 + 
        df['heart_disease'] * 3 + 
        df['is_diabetic_range'] * 2 +
        df['is_obese'] * 1
    )
    
    # Lifestyle risk score
    smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 1}
    df['smoking_numeric'] = df['smoking_status'].map(smoking_map) if 'smoking_status' in df.columns else 0
    df['lifestyle_risk'] = df['smoking_numeric'] + df['is_obese']
    
    return df

def prepare_data():
    """Load and preprocess data"""
    print("=" * 80)
    print("STEP 1: Loading and Preprocessing Data")
    print("=" * 80)
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    test_ids = test_df['id']
    
    # Combine for consistent preprocessing
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['stroke'] = -1
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # Handle missing BMI
    combined['bmi'] = combined['bmi'].fillna(combined['bmi'].median())
    
    # Feature Engineering
    print("Creating advanced features...")
    combined = advanced_feature_engineering(combined)
    
    # One-Hot Encoding
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    
    # Drop ID
    combined = combined.drop('id', axis=1, errors='ignore')
    
    # Split back
    X_train = combined[combined['is_train'] == 1].drop(['stroke', 'is_train'], axis=1)
    y_train = combined[combined['is_train'] == 1]['stroke']
    X_test = combined[combined['is_train'] == 0].drop(['stroke', 'is_train'], axis=1)
    
    print(f"Training shape: {X_train.shape}")
    print(f"Class distribution: {y_train.value_counts(normalize=True).to_dict()}")
    
    return X_train, y_train, X_test, test_ids

def train_lightgbm(X_train, y_train):
    """Train LightGBM model"""
    print("\n" + "=" * 80)
    print("STEP 2: Training LightGBM")
    print("=" * 80)
    
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("Installing LightGBM...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'lightgbm'])
        from lightgbm import LGBMClassifier
    
    # Calculate scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_weight = neg / pos
    
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=7,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print(f"Cross-Validation AUC: {cv_scores.mean():.5f} (+/- {cv_scores.std():.5f})")
    print(f"Individual Fold Scores: {[f'{s:.5f}' for s in cv_scores]}")
    
    # Train on full data
    model.fit(X_train, y_train)
    
    return model, cv_scores.mean()

def train_catboost(X_train, y_train):
    """Train CatBoost model"""
    print("\n" + "=" * 80)
    print("STEP 3: Training CatBoost")
    print("=" * 80)
    
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("Installing CatBoost...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'catboost'])
        from catboost import CatBoostClassifier
    
    # Calculate scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_weight = neg / pos
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.01,
        depth=6,
        l2_leaf_reg=3,
        subsample=0.8,
        colsample_bylevel=0.8,
        scale_pos_weight=scale_weight,
        random_state=RANDOM_STATE,
        verbose=False,
        thread_count=-1
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print(f"Cross-Validation AUC: {cv_scores.mean():.5f} (+/- {cv_scores.std():.5f})")
    print(f"Individual Fold Scores: {[f'{s:.5f}' for s in cv_scores]}")
    
    # Train on full data
    model.fit(X_train, y_train, verbose=False)
    
    return model, cv_scores.mean()

def create_super_ensemble(models, model_scores, X_test):
    """Create weighted ensemble based on CV scores"""
    print("\n" + "=" * 80)
    print("STEP 4: Creating Super Ensemble")
    print("=" * 80)
    
    # Calculate weights based on CV scores (higher score = higher weight)
    total_score = sum(model_scores)
    weights = [score / total_score for score in model_scores]
    
    print(f"Ensemble Weights:")
    for i, (name, weight) in enumerate(zip(['LightGBM', 'CatBoost'], weights)):
        print(f"  {name}: {weight:.3f}")
    
    # Get predictions
    predictions = []
    for model in models:
        pred = model.predict_proba(X_test)[:, 1]
        predictions.append(pred)
    
    # Weighted average
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    
    return ensemble_pred

def main():
    print("\n" + "=" * 80)
    print("ADVANCED GRADIENT BOOSTING TRAINING")
    print("Models: LightGBM + CatBoost")
    print("Target: AUC 0.85+")
    print("=" * 80 + "\n")
    
    # Step 1: Prepare Data
    X_train, y_train, X_test, test_ids = prepare_data()
    
    # Step 2: Train LightGBM
    lgbm_model, lgbm_score = train_lightgbm(X_train, y_train)
    
    # Step 3: Train CatBoost
    catboost_model, catboost_score = train_catboost(X_train, y_train)
    
    # Step 4: Create Super Ensemble
    ensemble_pred = create_super_ensemble(
        [lgbm_model, catboost_model],
        [lgbm_score, catboost_score],
        X_test
    )
    
    # Step 5: Save Submissions
    print("\n" + "=" * 80)
    print("STEP 5: Saving Submissions")
    print("=" * 80)
    
    # Individual model submissions
    lgbm_pred = lgbm_model.predict_proba(X_test)[:, 1]
    catboost_pred = catboost_model.predict_proba(X_test)[:, 1]
    
    # Save LightGBM
    pd.DataFrame({'id': test_ids, 'stroke': lgbm_pred}).to_csv(
        f"{SUBMISSION_DIR}/submission_lightgbm.csv", index=False
    )
    print(f"✅ Saved: submission_lightgbm.csv (CV AUC: {lgbm_score:.5f})")
    
    # Save CatBoost
    pd.DataFrame({'id': test_ids, 'stroke': catboost_pred}).to_csv(
        f"{SUBMISSION_DIR}/submission_catboost.csv", index=False
    )
    print(f"✅ Saved: submission_catboost.csv (CV AUC: {catboost_score:.5f})")
    
    # Save Ensemble
    pd.DataFrame({'id': test_ids, 'stroke': ensemble_pred}).to_csv(
        f"{SUBMISSION_DIR}/submission_lgbm_catboost_ensemble.csv", index=False
    )
    print(f"✅ Saved: submission_lgbm_catboost_ensemble.csv (Expected AUC: {max(lgbm_score, catboost_score):.5f}+)")
    
    # Save models
    joblib.dump(lgbm_model, f"{MODEL_DIR}/model_lightgbm.pkl")
    joblib.dump(catboost_model, f"{MODEL_DIR}/model_catboost.pkl")
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel Performance:")
    print(f"  LightGBM CV AUC: {lgbm_score:.5f}")
    print(f"  CatBoost CV AUC: {catboost_score:.5f}")
    print(f"  Best Single Model: {'LightGBM' if lgbm_score > catboost_score else 'CatBoost'}")
    print(f"\nRecommended Submission:")
    if lgbm_score > catboost_score:
        print(f"  🎯 submission_lightgbm.csv (Expected Kaggle: ~{lgbm_score:.3f})")
    else:
        print(f"  🎯 submission_catboost.csv (Expected Kaggle: ~{catboost_score:.3f})")
    print(f"  Alternative: submission_lgbm_catboost_ensemble.csv")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
