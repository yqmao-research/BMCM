import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_acc(y_true, y_pred):
    d = int(max(y_pred.max(), y_true.max())) + 1
    w = np.zeros((d, d), dtype=np.int64)
    np.add.at(w, (y_pred, y_true), 1)
    r, c = linear_sum_assignment(w.max() - w)
    return w[r, c].sum() / y_pred.size


def evaluate(y_true, y_pred):
    return {
        "acc": float(cluster_acc(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
    }
