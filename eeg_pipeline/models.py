from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_models():
    return {
        'SVC': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'RandomForest': RandomForestClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'XGB': XGBClassifier(random_state=42),
        'LGBM': LGBMClassifier(random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
    }
