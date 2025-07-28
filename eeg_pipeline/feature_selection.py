import numpy as np
from sklearn.linear_model import ElasticNet

def select_features(X_train, y_train, X_test, top_k=20):
    en = ElasticNet(random_state=42)
    en.fit(X_train, y_train)
    selected = X_train.columns[np.abs(en.coef_) > 0][:top_k]
    return X_train[selected], X_test[selected], selected
