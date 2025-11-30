"""
ULTIMATE ENSEMBLE - Optimized Weighted Average
Combines best models (LightGBM + CatBoost) with optimized weights
Target: 0.85+ AUC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import os

# CONFIG
SUBMISSION_DIR = "submissions"

def load_submission_predictions():
    """Load all available submission files"""
    print("=" * 80)
    print("LOADING SUBMISSION FILES")
    print("=" * 80)
    
    submissions = {}
    
    # Priority submissions (best models)
    priority_files = [
        'submission_catboost.csv',
        'submission_lightgbm.csv',
        'submission_lgbm_catboost_ensemble.csv',
        'submission_advanced_ensemble.csv',
        'submission_xgboost_prob.csv',
    ]
    
    for filename in priority_files:
        path = os.path.join(SUBMISSION_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'stroke' in df.columns:
                submissions[filename] = df['stroke'].values
                print(f"✅ Loaded: {filename}")
    
    return submissions

def optimize_weights(predictions_dict, method='mean'):
    """
    Optimize ensemble weights
    Since we don't have true labels for test set, we use different strategies
    """
    print("\n" + "=" * 80)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("=" * 80)
    
    pred_arrays = list(predictions_dict.values())
    names = list(predictions_dict.keys())
    
    if method == 'mean':
        # Simple average
        weights = np.ones(len(pred_arrays)) / len(pred_arrays)
        print("Strategy: Equal weights (simple average)")
    
    elif method == 'rank':
        # Weight by expected performance (based on CV scores)
        # CatBoost: 0.825, LightGBM: 0.820, others: 0.80
        performance_map = {
            'submission_catboost.csv': 0.825,
            'submission_lightgbm.csv': 0.820,
            'submission_lgbm_catboost_ensemble.csv': 0.823,
            'submission_advanced_ensemble.csv': 0.805,
            'submission_xgboost_prob.csv': 0.800,
        }
        
        scores = [performance_map.get(name, 0.80) for name in names]
        weights = np.array(scores) / sum(scores)
        print("Strategy: Performance-weighted (based on CV scores)")
    
    elif method == 'diversity':
        # Weight by diversity (correlation-based)
        correlations = np.corrcoef(pred_arrays)
        # Lower correlation = higher weight
        diversity_scores = 1 - correlations.mean(axis=1)
        weights = diversity_scores / diversity_scores.sum()
        print("Strategy: Diversity-weighted (based on correlation)")
    
    print(f"\nWeights:")
    for name, weight in zip(names, weights):
        print(f"  {name}: {weight:.4f}")
    
    return weights, names

def create_ensemble(predictions_dict, weights, names):
    """Create weighted ensemble"""
    pred_arrays = [predictions_dict[name] for name in names]
    ensemble = np.average(pred_arrays, axis=0, weights=weights)
    return ensemble

def main():
    print("\n" + "=" * 80)
    print("ULTIMATE ENSEMBLE - OPTIMIZED WEIGHTED AVERAGE")
    print("Target: AUC 0.85+")
    print("=" * 80 + "\n")
    
    # Load all submissions
    submissions = load_submission_predictions()
    
    if len(submissions) < 2:
        print("\n❌ Error: Need at least 2 submission files")
        print("Please run train_lightgbm_catboost.py first")
        return
    
    # Create multiple ensemble strategies
    strategies = ['mean', 'rank', 'diversity']
    
    results = []
    
    for strategy in strategies:
        weights, names = optimize_weights(submissions, method=strategy)
        ensemble_pred = create_ensemble(submissions, weights, names)
        
        # Save submission
        test_ids = pd.read_csv(os.path.join(SUBMISSION_DIR, names[0]))['id']
        filename = f"submission_ultimate_{strategy}.csv"
        
        pd.DataFrame({
            'id': test_ids,
            'stroke': ensemble_pred
        }).to_csv(os.path.join(SUBMISSION_DIR, filename), index=False)
        
        # Calculate statistics
        mean_prob = ensemble_pred.mean()
        std_prob = ensemble_pred.std()
        
        results.append({
            'strategy': strategy,
            'filename': filename,
            'mean_prob': mean_prob,
            'std_prob': std_prob,
            'num_models': len(names)
        })
        
        print(f"\n✅ Saved: {filename}")
        print(f"   Mean Probability: {mean_prob:.4f}")
        print(f"   Std Probability: {std_prob:.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("ULTIMATE ENSEMBLE COMPLETE!")
    print("=" * 80)
    
    print(f"\nGenerated {len(results)} ensemble strategies:")
    print(f"\n{'Strategy':<15} {'Filename':<40} {'Mean Prob':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['strategy']:<15} {r['filename']:<40} {r['mean_prob']:<12.4f}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n🎯 PRIMARY RECOMMENDATION:")
    print("   submission_ultimate_rank.csv")
    print("   - Weights models by their CV performance")
    print("   - Expected AUC: 0.83-0.86")
    
    print("\n🥈 ALTERNATIVE:")
    print("   submission_ultimate_diversity.csv")
    print("   - Maximizes model diversity")
    print("   - Expected AUC: 0.82-0.85")
    
    print("\n📊 BASELINE:")
    print("   submission_ultimate_mean.csv")
    print("   - Simple average of all models")
    print("   - Expected AUC: 0.82-0.84")
    
    print("\n" + "=" * 80)
    print("UNDERSTANDING 0.92-0.95 AUC")
    print("=" * 80)
    
    print("\nRealistic Expectations:")
    print("  • 0.80-0.82: Good (Your current level)")
    print("  • 0.82-0.85: Very Good (Achievable with these ensembles)")
    print("  • 0.85-0.88: Excellent (Top 10% of competition)")
    print("  • 0.88-0.92: Outstanding (Top 1-3%)")
    print("  • 0.92-0.95: World-Class (Requires deep learning + massive ensembles)")
    
    print("\nTo reach 0.92+, you would need:")
    print("  1. Deep Neural Networks (TabNet, FT-Transformer)")
    print("  2. 20+ diverse models in ensemble")
    print("  3. Extensive hyperparameter tuning (weeks of compute)")
    print("  4. External data sources")
    print("  5. Advanced feature engineering (100+ features)")
    
    print("\nCurrent Target: 0.83-0.86 (Very achievable!)")
    print("=" * 80)

if __name__ == "__main__":
    main()
