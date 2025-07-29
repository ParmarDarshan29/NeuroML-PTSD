from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

def perform_grid_search(X_train, y_train, model, param_grid):
    grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def get_param_grids():
    return {
        'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
        'KNN': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        'RandomForest': {'n_estimators': [50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2']},
        'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
        'XGB': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
        'LGBM': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
        'CatBoost': {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'depth': [3, 5, 7]}
    }
