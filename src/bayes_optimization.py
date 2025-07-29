import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def rf_cv(n_estimators, max_features, X, y):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_features=max(min(max_features, 1.0), 0.1),  # ensure valid float range
        random_state=42
    )
    return cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()

def cb_cv(iterations, learning_rate, depth, X, y):
    model = CatBoostClassifier(
        iterations=int(iterations),
        learning_rate=learning_rate,
        depth=int(depth),
        verbose=0,
        random_state=42
    )
    return cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()

def lgbm_cv(n_estimators, learning_rate, X, y):
    model = LGBMClassifier(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        random_state=42,
        verbose=-1
    )
    return cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()

def xgb_cv(n_estimators, learning_rate, max_depth, X, y):
    model = XGBClassifier(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        random_state=42
    )
    return cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()

def run_all_bayesian_optimizations(X, y):
    results = []

    rf_opt = BayesianOptimization(
        f=lambda n_estimators, max_features: rf_cv(n_estimators, max_features, X, y),
        pbounds={'n_estimators': (50, 200), 'max_features': (0.1, 1.0)},
        random_state=42
    )
    rf_opt.maximize(init_points=10, n_iter=30)
    results.append({'Model': 'RandomForest', 'Best Parameters': rf_opt.max['params'], 'Best CV Score (ROC AUC)': rf_opt.max['target']})

    cb_opt = BayesianOptimization(
        f=lambda iterations, learning_rate, depth: cb_cv(iterations, learning_rate, depth, X, y),
        pbounds={'iterations': (50, 200), 'learning_rate': (0.01, 0.2), 'depth': (3, 7)},
        random_state=42
    )
    cb_opt.maximize(init_points=10, n_iter=30)
    results.append({'Model': 'CatBoost', 'Best Parameters': cb_opt.max['params'], 'Best CV Score (ROC AUC)': cb_opt.max['target']})

    lgbm_opt = BayesianOptimization(
        f=lambda n_estimators, learning_rate: lgbm_cv(n_estimators, learning_rate, X, y),
        pbounds={'n_estimators': (50, 200), 'learning_rate': (0.01, 0.2)},
        random_state=42
    )
    lgbm_opt.maximize(init_points=10, n_iter=30)
    results.append({'Model': 'LGBM', 'Best Parameters': lgbm_opt.max['params'], 'Best CV Score (ROC AUC)': lgbm_opt.max['target']})

    xgb_opt = BayesianOptimization(
        f=lambda n_estimators, learning_rate, max_depth: xgb_cv(n_estimators, learning_rate, max_depth, X, y),
        pbounds={'n_estimators': (50, 200), 'learning_rate': (0.01, 0.2), 'max_depth': (3, 7)},
        random_state=42
    )
    xgb_opt.maximize(init_points=10, n_iter=30)
    results.append({'Model': 'XGB', 'Best Parameters': xgb_opt.max['params'], 'Best CV Score (ROC AUC)': xgb_opt.max['target']})

    return pd.DataFrame(results)
