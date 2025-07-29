import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

def permutation_test(model, X, y, num_permutations=1000):
    original_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    permuted_aucs = [roc_auc_score(shuffle(y, random_state=i), model.predict_proba(X)[:, 1]) for i in range(num_permutations)]
    p_val = np.mean([auc > original_auc for auc in permuted_aucs])
    return p_val
