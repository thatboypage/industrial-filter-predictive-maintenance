import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from data import FAIL_DP, eol_time
DUSTS = ['A2', 'A3', 'A4']

def features(r, t_now):
    i = np.searchsorted(r['t'], t_now, side='right')
    t, dp, m = r['t'][:i], r['dp'][:i], r['m'][:i]
    w = max(10, int(5 / 0.1))                      # last 5 s window
    a, b = max(0, i - w), i
    slope_t = np.polyfit(t[a:b], dp[a:b], 1)[0] if b - a > 5 else 0.0
    w2 = max(10, int(15 / 0.1)); a2 = max(0, i - w2)
    slope_t15 = np.polyfit(t[a2:b], dp[a2:b], 1)[0] if b - a2 > 5 else 0.0
    return [DUSTS.index(r['dust']), r['Q'], r['feed'], r['mrate'], t_now, m[-1],
            dp[a:b].mean(), dp.max(), slope_t, slope_t15, slope_t / r['mrate'], slope_t15 / r['mrate']]

def make_training_set(runs, label='pseudo', step=0.5):
    X, y, g = [], [], []
    for n, r in runs.items():
        t_end = r['t'][-1]
        eol = eol_time(r)
        t_fail = eol if (label == 'true' and not np.isnan(eol)) else t_end   # pseudo: replacement treated as failure
        for tc in np.arange(5.0, min(t_end, t_fail) + 1e-9, step):
            X.append(features(r, tc)); y.append(t_fail - tc); g.append(n)
    return np.array(X), np.array(y), np.array(g)

class GBM:
    def __init__(self, label):
        self.label = label
    def fit(self, runs):
        X, y, _ = make_training_set(runs, self.label)
        self.m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, categorical_features=[0], random_state=0).fit(X, y)
        return self
    def predict(self, r, t_now):
        return max(0.0, float(self.m.predict(np.array([features(r, t_now)]))[0]))

class LinearTrend:
    """No training: extrapolate the unit's own last-15 s pressure trend to 600 Pa."""
    def predict(self, r, t_now):
        f = features(r, t_now); dp_now, s = f[6], f[9]
        if s <= 1e-6: return 300.0
        return float(np.clip((FAIL_DP - dp_now) / s, 0, 300))

class WeibullAFT:
    """Censoring-aware lifetime model on dust mass at failure (no sensor data).
    log M = b0 + b_dust + b_q*log(Q/60) + sigma*eps, eps ~ standard min-extreme-value.  RUL = E[M - m | M > m] / mrate."""
    def _X(self, r):
        # Dust type cannot be a covariate: all 5 observed failures are A2 dust, so A3/A4 effects are unidentifiable
        return np.array([1.0, np.log(r['Q'] / 60.0)], float)
    def fit(self, runs):
        X, M, E = [], [], []
        for r in runs.values():
            e = eol_time(r)
            X.append(self._X(r)); E.append(not np.isnan(e))
            M.append(r['mrate'] * (e if E[-1] else r['t'][-1]))
        X, M, E = np.array(X), np.array(M), np.array(E)
        def nll(th):
            b, s = th[:-1], np.exp(th[-1]); z = (np.log(M) - X @ b) / s
            return -np.sum(E * (z - np.log(s) - np.log(M)) - np.exp(z))
        th0 = np.r_[np.log(M.mean()) + 0.5, 0, np.log(0.3)]
        self.th = minimize(nll, th0, method='Nelder-Mead', options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10)).x
        self.n_events = int(E.sum()); return self
    def predict(self, r, t_now):
        b, s = self.th[:-1], np.exp(self.th[-1]); mu = self._X(r) @ b
        m_now = r['mrate'] * t_now
        grid = np.linspace(m_now, m_now + 200, 20001)
        S = np.exp(-np.exp((np.log(grid) - mu) / s))
        mrl = np.trapezoid(S, grid) / max(S[0], 1e-300)
        return float(mrl / r['mrate'])
