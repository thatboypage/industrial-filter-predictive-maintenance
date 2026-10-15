"""Label-free validation on the (censored) training set only.
For each training run (leave-one-run-out), cut the history at 40/60/80% of its observed length and
forecast when the pressure reaches the last observed value. The true crossing time is known even though
the run is censored, so this checks the extrapolation mechanism without touching the test set."""
import sys, numpy as np
sys.path.insert(0, 'src')
from data import load
from physics import PhysicsModel, inv_softplus
tr = load('Train_data.csv')
def t_to_threshold(pm, r, t_cut, p_thr):
    z, _ = pm.unit_posterior(r, t_cut); p0 = pm.p0[r['dust']]; K = pm.K(r['dust'], r['Q'], z[1])
    m_thr = z[0] + p0 / K * inv_softplus(p_thr / p0)
    return max(m_thr - r['mrate'] * t_cut, 0) / r['mrate']
PRIOR = sys.argv[1] == '1'
for sd in [float(a) for a in sys.argv[2:]] or [0.15, 0.25, 0.35, 0.5, 0.75]:
    errs = []
    for n, r in tr.items():
        others = {k: v for k, v in tr.items() if k != n}
        pm = PhysicsModel(sd_logk=sd, m0_prior=PRIOR).fit(others)
        p_end = np.convolve(r['dp'], np.ones(10) / 10, 'valid')[-1]
        if p_end < 80: continue
        t_cross = r['t'][np.argmax(np.convolve(r['dp'], np.ones(10)/10, 'same') >= p_end - 1)]
        for f in [0.4, 0.6, 0.8]:
            tc = f * t_cross
            errs.append(t_to_threshold(pm, r, tc, p_end) - (t_cross - tc))
    e = np.array(errs)
    print(f'sd_logk={sd}: n={len(e)} RMSE={np.sqrt(np.mean(e**2)):.2f} MAE={np.mean(abs(e)):.2f} bias={e.mean():.2f}', flush=True)
