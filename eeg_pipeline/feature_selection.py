import numpy as np
from sklearn.linear_model import ElasticNet

def select_top_features(X_train, y_train, top_k=20):
    en = ElasticNet(random_state=42)
    en.fit(X_train, y_train)
    selected_features = X_train.columns[np.abs(en.coef_) > 0]
    return selected_features[:top_k]
