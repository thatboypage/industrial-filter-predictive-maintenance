import numpy as np
def score_phm(d):          # PHM08-style asymmetric score: late (over-)predictions punished harder
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
def metrics(pred, true):
    pred, true = np.asarray(pred), np.asarray(true); d = pred - true
    return dict(RMSE=np.sqrt(np.mean(d**2)), MAE=np.mean(np.abs(d)),
                MedAPE=np.median(np.abs(d) / true) * 100, late_pct=np.mean(d > 0) * 100,
                late_gt10=np.mean(d > 10) * 100, PHM=np.sum(score_phm(d)))
def bootstrap_ci(pred, true, stat='RMSE', B=4000, seed=0):
    rng = np.random.default_rng(seed); pred, true = np.asarray(pred), np.asarray(true); n = len(true); v = []
    for _ in range(B):
        i = rng.integers(0, n, n); v.append(metrics(pred[i], true[i])[stat])
    return np.percentile(v, [2.5, 97.5])
