"""Calibration audit on existing OOF predictions; no model retraining."""
import numpy as np


def isotonic(train_x, train_y, test_x):
    x, inv = np.unique(np.asarray(train_x, float), return_inverse=True)
    weights = np.bincount(inv).astype(float)
    means = np.bincount(inv, weights=np.asarray(train_y, float)) / weights
    blocks = []
    for i, (v, w) in enumerate(zip(means, weights)):
        blocks.append([i, i, v, w])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            a, b = blocks[-2:]
            w = a[3] + b[3]
            blocks[-2:] = [[a[0], b[1], (a[2]*a[3]+b[2]*b[3])/w, w]]
    fitted = np.empty(len(x))
    for a, b, v, _ in blocks:
        fitted[a:b+1] = v
    return np.interp(test_x, x, fitted)


def platt(train_x, train_y, test_x):
    def design(x):
        p = np.clip(np.asarray(x, float), 1e-6, 1-1e-6)
        return np.column_stack([np.ones(len(p)), np.log(p/(1-p))])
    X, y = design(train_x), np.asarray(train_y, float)
    rate = np.clip(y.mean(), 1e-6, 1-1e-6)
    beta = np.array([np.log(rate/(1-rate)), 0.0])
    for _ in range(100):
        p = 1/(1+np.exp(-np.clip(X@beta, -35, 35)))
        gradient = X.T@(p-y) + np.array([0, 1e-4*beta[1]])
        hessian = X.T@((p*(1-p))[:, None]*X) + np.diag([1e-8, 1e-4])
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    if beta[1] < 0:
        raise ValueError('Negative calibration slope: do not deploy')
    return 1/(1+np.exp(-np.clip(design(test_x)@beta, -35, 35)))


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p*10).astype(int), 9)
    ece = sum(np.sum(bins==b)/len(p)*abs(p[bins==b].mean()-y[bins==b].mean())
              for b in np.unique(bins))
    q = np.clip(p, 1e-10, 1-1e-10)
    return {'brier': float(np.mean((p-y)**2)), 'ece_10_fixed_bins': float(ece),
            'log_loss': float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q)))}
