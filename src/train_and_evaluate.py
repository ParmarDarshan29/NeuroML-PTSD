# eeg_pipeline/train_and_evaluate.py

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from .feature_selection import select_features
from .model_registry import get_models, get_param_grids
from .config import N_SPLITS, RANDOM_STATE
from .bayesian_optimization import optimize_with_bayes

def cross_validate_and_train(X, y):
    models = get_models()
    param_grids = get_param_grids()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    auc_scores, best_models, best_params, p_scores = {}, {}, {}, {}

    for name in models:
        auc_scores[name] = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_sel, X_test_sel, selected = select_features(X_train, y_train, X_test)

        for name, model in models.items():
            grid = GridSearchCV(model, param_grids[name], scoring='roc_auc', cv=3)
            grid.fit(X_train_sel, y_train)
            y_pred = grid.predict_proba(X_test_sel)[:, 1]

            auc_scores[name].append(roc_auc_score(y_test, y_pred))
            best_models[name] = grid.best_estimator_
            best_params[name] = grid.best_params_

    return auc_scores, best_models, best_params, selected

def permutation_test(model, X, y, n_permutations=1000):
    original = roc_auc_score(y, model.predict_proba(X)[:, 1])
    scores = []

    for _ in range(n_permutations):
        y_perm = shuffle(y, random_state=np.random.randint(0, 10000))
        score = roc_auc_score(y_perm, model.predict_proba(X)[:, 1])
        scores.append(score)

    return np.mean([s > original for s in scores])

def run_bayesian_optimization(X, y, selected_features):
    return optimize_with_bayes(X, y, selected_features)
