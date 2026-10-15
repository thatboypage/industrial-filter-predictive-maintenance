import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from data import load, eol_time
from physics import PhysicsModel
from baselines import GBM, LinearTrend, WeibullAFT, features, DUSTS
from sklearn.ensemble import HistGradientBoostingRegressor
from evaluate import metrics, bootstrap_ci

tr  = load('Train_data.csv')               # what a practitioner has: right-censored lives
tru = load('Train_data_uncensored.csv')    # oracle only: the same 50 filters run to failure
te  = load('Test_data.csv')
SD = 0.06                                  # chosen by label-free validation on training data only

class ImputedGBM:
    """Hybrid: physics model imputes each censored filter's failure time, then GBM learns RUL from sensors."""
    def fit(self, runs, pm):
        X, y = [], []
        self.imputed = {}
        for n, r in runs.items():
            e = eol_time(r)
            if np.isnan(e):
                e = r['t'][-1] + pm.predict_rul(r, r['t'][-1])[0]
            self.imputed[n] = e
            for tc in np.arange(5.0, r['t'][-1] + 1e-9, 0.5):
                X.append(features(r, tc)); y.append(e - tc)
        self.m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, categorical_features=[0], random_state=0).fit(np.array(X), np.array(y))
        return self
    def predict(self, r, t):
        return max(0.0, float(self.m.predict(np.array([features(r, t)]))[0]))

pm = PhysicsModel(sd_logk=SD).fit(tr)
pm_or = PhysicsModel(sd_logk=SD).fit(tru)
weib = WeibullAFT().fit(tr)
models = {
  'Lifetime survival model (Weibull AFT, no sensors)': weib.predict,
  'Own-trend extrapolation (no training)': LinearTrend().predict,
  'Gradient boosting, naive labels (replacement = failure)': GBM('pseudo').fit(tr).predict,
  'Hybrid: physics-imputed lives + gradient boosting': ImputedGBM().fit(tr, pm).predict,
  'Physics-informed degradation model': lambda r, t: pm.predict_rul(r, t)[0],
  'ORACLE gradient boosting (true failure labels)': GBM('true').fit(tru).predict,
  'ORACLE physics model (fit on full lives)': lambda r, t: pm_or.predict_rul(r, t)[0],
}
true = np.array([r['rul_true'] for r in te.values()])
rows, preds = [], {}
for name, f in models.items():
    p = np.array([f(r, r['t'][-1]) for r in te.values()]); preds[name] = p
    m = metrics(p, true); lo, hi = bootstrap_ci(p, true)
    rows.append(dict(model=name, **{k: round(v, 2) for k, v in m.items()}, RMSE_CI=f'{lo:.1f}-{hi:.1f}'))
R = pd.DataFrame(rows); print(R.to_string(index=False))
# uncertainty calibration of the physics model
cov = []; ints = []
for r in te.values():
    _, (lo, med, hi) = pm.predict_rul(r, r['t'][-1]); cov.append(lo <= r['rul_true'] <= hi); ints.append((lo, med, hi))
print('Physics 90% interval coverage:', np.mean(cov))
print('Weibull events in training:', weib.n_events)
P = pd.DataFrame(preds); P['true'] = true; P['no'] = list(te.keys())
P['dust'] = [r['dust'] for r in te.values()]; P['dp_now'] = [r['dp'][-1] for r in te.values()]
P['pi_lo'] = [i[0] for i in ints]; P['pi_hi'] = [i[2] for i in ints]
P.to_csv('results/test_predictions.csv', index=False); R.to_csv('results/test_metrics.csv', index=False)
json.dump(dict(c=pm.c, p0=pm.p0, beta=pm.beta, m0_mu=pm.m0_mu, m0_sd=pm.m0_sd, coverage90=float(np.mean(cov))),
          open('results/physics_params.json', 'w'), indent=1, default=float)
