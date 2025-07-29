import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_optuna_param_spaces():
    return {
        "RandomForest": {
            "model": RandomForestClassifier,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 5, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            },
        },
        "XGBoost": {
            "model": XGBClassifier,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            },
        },
        "LGBM": {
            "model": LGBMClassifier,
            "params": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            },
        },
        "CatBoost": {
            "model": CatBoostClassifier,
            "params": lambda trial: {
                "iterations": trial.suggest_int("iterations", 50, 200),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            },
        },
    }

def optimize_with_optuna(X, y, n_trials=30):
    results = []
    search_space = get_optuna_param_spaces()

    for model_name, details in search_space.items():
        def objective(trial):
            model = details["model"](**details["params"](trial))
            scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
            return scores.mean()

        print(f"🔍 Optimizing {model_name} with Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_model = details["model"](**best_params)
        best_model.fit(X, y)

        results.append({
            "Model": model_name,
            "Best Parameters": best_params,
            "Best CV Score (ROC AUC)": study.best_value
        })

    return pd.DataFrame(results)
