import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Project imports
from src.data_loader import load_and_filter_data
from src.preprocessing import split_features_labels
from src.feature_selection import select_top_features
from src.bayes_optimization import run_all_bayesian_optimizations
from src.optuna_optimization import optimize_with_optuna
from src.xai_interpretation import (
    explain_with_shap,
    explain_with_lime,
    plot_permutation_importance,
    plot_model_feature_importance,
    plot_elasticnet_weights
)

from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

model_map = {
    "RandomForest": RandomForestClassifier,
    "CatBoost": CatBoostClassifier,
    "LGBM": LGBMClassifier,
    "XGB": XGBClassifier,
}

# ========================================
# 1. Load + preprocess data
# ========================================
data = load_and_filter_data()
X, y = split_features_labels(data)

# ========================================
# 2. Feature selection
# ========================================
X = X.copy()
selected_features = select_top_features(X, y, top_k=20)
X_selected = X[selected_features]

# ========================================
# 3. Run both optimization methods
# ========================================
print("🔍 Running Bayesian Optimization...")
bayes_results = run_all_bayesian_optimizations(X_selected, y)
print("✅ Done.\n")

print("🔍 Running Optuna Optimization...")
optuna_results = optimize_with_optuna(X_selected, y, n_trials=30)
print("✅ Done.\n")

combined = pd.concat([bayes_results, optuna_results], ignore_index=True)

# ========================================
# 4. Find best model across all results
# ========================================
best_model_info = combined.loc[combined['Best CV Score (ROC AUC)'].idxmax()]
best_model_name = best_model_info['Model']
best_params = best_model_info['Best Parameters']

# Cast types
if best_model_name == "RandomForest":
    best_params["n_estimators"] = int(best_params["n_estimators"])
    if isinstance(best_params.get("max_features", 1), float) and best_params["max_features"] > 1:
        best_params["max_features"] = float(best_params["max_features"])
elif best_model_name == "CatBoost":
    best_params["iterations"] = int(best_params["iterations"])
    best_params["depth"] = int(best_params["depth"])
elif best_model_name == "LGBM":
    best_params["n_estimators"] = int(best_params["n_estimators"])
elif best_model_name == "XGB":
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["max_depth"] = int(best_params["max_depth"])

print(f"🏆 Best Model: {best_model_name}")
print("🔧 Best Parameters:", best_params)

# ========================================
# 5. Cross-validated training & evaluation
# ========================================
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []
final_model = model_map[best_model_name](**best_params)

for train_idx, test_idx in kf.split(X_selected, y):
    X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    final_model.fit(X_train, y_train)
    y_pred = final_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    auc_scores.append(auc)

print(f"\n🎯 Mean ROC AUC (10-fold): {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")



import os

# Create results directory
os.makedirs("results", exist_ok=True)

# Save selected features
with open("results/selected_features.txt", "w") as f:
    f.write("\n".join(selected_features))

# Save optimization results
bayes_results.to_csv("results/bayes_results.csv", index=False)
optuna_results.to_csv("results/optuna_results.csv", index=False)
combined.to_csv("results/final_summary.csv", index=False)

# Save final cross-val AUC
with open("results/final_auc.txt", "w") as f:
    f.write(f"Mean AUC: {np.mean(auc_scores):.4f}\n")
    f.write(f"Std AUC: {np.std(auc_scores):.4f}\n")
    f.write(f"Model: {best_model_name}\n")
    f.write(f"Params: {best_params}\n")


# ========================================
# 6. Fit on full data and run XAI
# ========================================
final_model.fit(X_selected, y)

print("\n📊 SHAP Summary Plot:")
explain_with_shap(final_model, X_selected, X_selected)

print("\n📊 Model Feature Importances:")
plot_model_feature_importance(final_model, selected_features)

print("\n📊 Permutation Importances:")
plot_permutation_importance(final_model, X_selected, y, selected_features)

print("\n📊 ElasticNet Feature Weights:")
plot_elasticnet_weights(X_selected, y)

print("\n📊 LIME Explanation (1 sample):")
explain_with_lime(final_model, X_selected, X_selected, feature_names=selected_features)


