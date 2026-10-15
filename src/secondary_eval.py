"""(1) Conformal-style calibration of the physics model's uncertainty using label-free LORO residuals on training.
(2) Secondary evaluation: leave-one-filter-out on the 50 training filters, scored against their true failure
    times (from the uncensored release) at 25/50/75% of life. Models are fit only on the other 49 CENSORED filters."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from data import load, eol_time
from physics import PhysicsModel, inv_softplus
from baselines import GBM
from evaluate import metrics
tr, tru, te = load('Train_data.csv'), load('Train_data_uncensored.csv'), load('Test_data.csv')
SD = 0.06
calib, sec = [], []
for n, r in tr.items():
    others = {k: v for k, v in tr.items() if k != n}
    pm = PhysicsModel(sd_logk=SD).fit(others)
    # (1) label-free residuals: forecast time to reach this run's last observed pressure
    sm = np.convolve(r['dp'], np.ones(10) / 10, 'same'); p_end = sm[-6]
    if p_end >= 80:
        t_cross = r['t'][np.argmax(sm >= p_end - 1)]
        for f in [0.4, 0.6, 0.8]:
            tc = f * t_cross; z, _ = pm.unit_posterior(r, tc); p0 = pm.p0[r['dust']]
            m_thr = z[0] + p0 / pm.K(r['dust'], r['Q'], z[1]) * inv_softplus(p_end / p0)
            pred = max(m_thr - r['mrate'] * tc, 0) / r['mrate']
            calib.append(np.log((t_cross - tc + 1) / (pred + 1)))
    # (2) true-life evaluation on the held-out filter
    ru = tru[n]; eol = eol_time(ru); gbm = GBM('pseudo').fit(others)
    for f in [0.25, 0.5, 0.75]:
        tc = f * eol
        sec.append(dict(no=n, frac=f, true=eol - tc, physics=pm.predict_rul(ru, tc)[0], gbm_naive=gbm.predict(ru, tc)))
    print(n, end=' ', flush=True)
calib = np.array(calib); q = np.percentile(calib, [5, 95])
S = pd.DataFrame(sec); S.to_csv('results/secondary_loo.csv', index=False)
print('\nlog-ratio quantiles', q)
for f in [0.25, 0.5, 0.75, None]:
    d = S if f is None else S[S.frac == f]
    print(f, 'physics', {k: round(v, 2) for k, v in metrics(d.physics, d.true).items()},
          '\n     gbm', {k: round(v, 2) for k, v in metrics(d.gbm_naive, d.true).items()})
P = pd.read_csv('results/test_predictions.csv'); p = P['Physics-informed degradation model']
P['ci_lo'] = (p + 1) * np.exp(q[0]) - 1; P['ci_hi'] = (p + 1) * np.exp(q[1]) - 1
cov = ((P.true >= P.ci_lo) & (P.true <= P.ci_hi)).mean()
print('calibrated 90% interval coverage on test:', cov, 'median width', np.median(P.ci_hi - P.ci_lo))
P.to_csv('results/test_predictions.csv', index=False)
json.dump(dict(logratio_q05=q[0], logratio_q95=q[1], test_coverage90=cov), open('results/calibration.json', 'w'), default=float)
