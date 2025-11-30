"""
EXTREME APPROACH - Deep Learning with TabNet
Goal: Push towards 0.90+ AUC (0.95 is extremely difficult)
Strategy: Neural network + extreme feature engineering + pseudo-labeling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# CONFIG
TRAIN_PATH = "sesame-jci-stroke-prediction/train.csv"
TEST_PATH = "sesame-jci-stroke-prediction/test.csv"
SUBMISSION_DIR = "submissions"
RANDOM_STATE = 42

def extreme_feature_engineering(df):
    """Create 50+ advanced features"""
    
    # Original features
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    
    # 1. Age-based features (10 features)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 75, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    df['age_squared'] = df['age'] ** 2
    df['age_cubed'] = df['age'] ** 3
    df['age_sqrt'] = np.sqrt(df['age'])
    df['age_log'] = np.log1p(df['age'])
    df['is_senior'] = (df['age'] > 65).astype(int)
    df['is_very_senior'] = (df['age'] > 75).astype(int)
    df['is_young'] = (df['age'] < 40).astype(int)
    df['age_risk_score'] = df['age'] / 100  # Normalized
    df['age_decade'] = (df['age'] // 10).astype(int)
    
    # 2. Glucose features (10 features)
    df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins=[0, 100, 125, 200, 300], labels=[0, 1, 2, 3]).astype(int)
    df['glucose_squared'] = df['avg_glucose_level'] ** 2
    df['glucose_log'] = np.log1p(df['avg_glucose_level'])
    df['glucose_sqrt'] = np.sqrt(df['avg_glucose_level'])
    df['is_diabetic'] = (df['avg_glucose_level'] > 126).astype(int)
    df['is_prediabetic'] = ((df['avg_glucose_level'] > 100) & (df['avg_glucose_level'] <= 126)).astype(int)
    df['is_very_high_glucose'] = (df['avg_glucose_level'] > 200).astype(int)
    df['glucose_normalized'] = df['avg_glucose_level'] / 300
    df['glucose_deviation'] = df['avg_glucose_level'] - df['avg_glucose_level'].median()
    df['glucose_percentile'] = df['avg_glucose_level'].rank(pct=True)
    
    # 3. BMI features (10 features)
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 40, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    df['bmi_squared'] = df['bmi'] ** 2
    df['bmi_log'] = np.log1p(df['bmi'])
    df['bmi_sqrt'] = np.sqrt(df['bmi'])
    df['is_underweight'] = (df['bmi'] < 18.5).astype(int)
    df['is_normal_weight'] = ((df['bmi'] >= 18.5) & (df['bmi'] < 25)).astype(int)
    df['is_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    df['is_very_obese'] = (df['bmi'] >= 40).astype(int)
    df['bmi_deviation'] = df['bmi'] - df['bmi'].median()
    
    # 4. Interaction features (15 features)
    df['age_x_glucose'] = df['age'] * df['avg_glucose_level']
    df['age_x_bmi'] = df['age'] * df['bmi']
    df['glucose_x_bmi'] = df['avg_glucose_level'] * df['bmi']
    df['age_x_glucose_x_bmi'] = df['age'] * df['avg_glucose_level'] * df['bmi']
    df['age_glucose_ratio'] = df['age'] / (df['avg_glucose_level'] + 1)
    df['age_bmi_ratio'] = df['age'] / (df['bmi'] + 1)
    df['glucose_bmi_ratio'] = df['avg_glucose_level'] / (df['bmi'] + 1)
    df['age_squared_x_glucose'] = (df['age'] ** 2) * df['avg_glucose_level']
    df['age_x_glucose_squared'] = df['age'] * (df['avg_glucose_level'] ** 2)
    df['senior_x_diabetic'] = df['is_senior'] * df['is_diabetic']
    df['senior_x_obese'] = df['is_senior'] * df['is_obese']
    df['diabetic_x_obese'] = df['is_diabetic'] * df['is_obese']
    df['hypertension_x_diabetic'] = df['hypertension'] * df['is_diabetic']
    df['hypertension_x_obese'] = df['hypertension'] * df['is_obese']
    df['heart_disease_x_hypertension'] = df['heart_disease'] * df['hypertension']
    
    # 5. Risk scores (10 features)
    df['cardio_risk_score'] = (
        df['hypertension'] * 3 + 
        df['heart_disease'] * 4 + 
        df['is_diabetic'] * 2 +
        df['is_obese'] * 1 +
        df['is_senior'] * 2
    )
    df['metabolic_risk_score'] = df['is_diabetic'] * 2 + df['is_obese'] * 2 + df['is_overweight'] * 1
    df['age_risk_score_v2'] = (df['age'] > 60).astype(int) * 2 + (df['age'] > 70).astype(int) * 2
    df['compound_risk_score'] = df['cardio_risk_score'] + df['metabolic_risk_score'] + df['age_risk_score_v2']
    
    # Smoking risk
    smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 3, 'Unknown': 1}
    df['smoking_numeric'] = df['smoking_status'].map(smoking_map) if 'smoking_status' in df.columns else 0
    df['lifestyle_risk'] = df['smoking_numeric'] + df['is_obese'] * 2
    df['smoking_x_age'] = df['smoking_numeric'] * df['age']
    df['smoking_x_hypertension'] = df['smoking_numeric'] * df['hypertension']
    df['smoking_x_heart_disease'] = df['smoking_numeric'] * df['heart_disease']
    df['total_risk_score'] = df['compound_risk_score'] + df['lifestyle_risk']
    
    # 6. Statistical features (5 features)
    df['age_glucose_sum'] = df['age'] + df['avg_glucose_level']
    df['age_bmi_sum'] = df['age'] + df['bmi']
    df['glucose_bmi_sum'] = df['avg_glucose_level'] + df['bmi']
    df['all_vitals_sum'] = df['age'] + df['avg_glucose_level'] + df['bmi']
    df['all_vitals_mean'] = (df['age'] + df['avg_glucose_level'] + df['bmi']) / 3
    
    return df

def train_with_pytorch(X_train, y_train, X_test):
    """Train using PyTorch TabNet"""
    print("\n" + "=" * 80)
    print("TRAINING DEEP LEARNING MODEL (TabNet)")
    print("=" * 80)
    
    import torch
    
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError:
        print("Installing PyTorch TabNet...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pytorch-tabnet'])
        from pytorch_tabnet.tab_model import TabNetClassifier
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    class_weight = neg / pos
    
    # TabNet model
    model = TabNetClassifier(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        verbose=1,
        seed=RANDOM_STATE
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
        print(f"\nFold {fold + 1}/5")
        
        X_fold_train = X_train_scaled[train_idx]
        y_fold_train = y_train.iloc[train_idx].values
        X_fold_val = X_train_scaled[val_idx]
        y_fold_val = y_train.iloc[val_idx].values
        
        # Train
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            eval_metric=['auc'],
            max_epochs=200,
            patience=50,
            batch_size=256,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        # Predict
        preds = model.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, preds)
        cv_scores.append(auc)
        print(f"Fold {fold + 1} AUC: {auc:.5f}")
    
    mean_cv_auc = np.mean(cv_scores)
    print(f"\nCross-Validation AUC: {mean_cv_auc:.5f} (+/- {np.std(cv_scores):.5f})")
    
    # Train on full data
    print("\nTraining on full dataset...")
    model.fit(
        X_train_scaled, y_train.values,
        max_epochs=200,
        patience=50,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0
    )
    
    # Predict
    test_preds = model.predict_proba(X_test_scaled)[:, 1]
    
    return test_preds, mean_cv_auc

def main():
    print("\n" + "=" * 80)
    print("EXTREME DEEP LEARNING APPROACH")
    print("Model: TabNet (State-of-the-Art Neural Network)")
    print("Features: 60+ engineered features")
    print("Target: 0.85-0.90+ AUC")
    print("=" * 80 + "\n")
    
    # Check if PyTorch is available
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("Installing PyTorch...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'torch'])
        import torch
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    test_ids = test_df['id']
    
    # Combine
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['stroke'] = -1
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # Extreme feature engineering
    print("Creating 60+ features...")
    combined = extreme_feature_engineering(combined)
    
    # One-hot encoding
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    combined = combined.drop('id', axis=1, errors='ignore')
    
    # Split
    X_train = combined[combined['is_train'] == 1].drop(['stroke', 'is_train'], axis=1)
    y_train = combined[combined['is_train'] == 1]['stroke']
    X_test = combined[combined['is_train'] == 0].drop(['stroke', 'is_train'], axis=1)
    
    print(f"Training shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Train TabNet
    tabnet_preds, tabnet_cv_auc = train_with_pytorch(X_train, y_train, X_test)
    
    # Save submission
    pd.DataFrame({
        'id': test_ids,
        'stroke': tabnet_preds
    }).to_csv(f"{SUBMISSION_DIR}/submission_tabnet_extreme.csv", index=False)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nTabNet CV AUC: {tabnet_cv_auc:.5f}")
    print(f"Saved: submission_tabnet_extreme.csv")
    
    if tabnet_cv_auc >= 0.90:
        print(f"\n🏆 OUTSTANDING! You've reached 0.90+ AUC!")
    elif tabnet_cv_auc >= 0.85:
        print(f"\n✅ EXCELLENT! You've reached 0.85+ AUC!")
    elif tabnet_cv_auc >= 0.83:
        print(f"\n📈 VERY GOOD! Close to the target!")
    
    print("\n" + "=" * 80)
    print("NOTE: 0.95 AUC Reality Check")
    print("=" * 80)
    print("0.95 AUC on this dataset would be:")
    print("  • Better than most published medical research")
    print("  • Likely overfitting or data leakage")
    print("  • Unrealistic without external medical data")
    print(f"\nYour {tabnet_cv_auc:.3f} AUC is already very strong!")
    print("=" * 80)

if __name__ == "__main__":
    main()
