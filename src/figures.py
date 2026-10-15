import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
from data import load
from physics import PhysicsModel, curve, inv_softplus
BLUE, ORANGE, GRAY, LGRAY, INK, INK2 = '#2a78d6', '#eb6834', '#8a8984', '#c9c8c2', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': LGRAY, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#ecebe7', 'grid.linewidth': 0.8, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
M = pd.read_csv('results/test_metrics.csv'); P = pd.read_csv('results/test_predictions.csv')
PH = 'Physics-informed degradation model'; GB = 'Gradient boosting, naive labels (replacement = failure)'
short = {'Lifetime survival model (Weibull AFT, no sensors)': 'Lifetime survival model (no sensors)',
         'Own-trend extrapolation (no training)': 'Linear trend extrapolation',
         GB: 'Gradient boosting, naive labels', 'Hybrid: physics-imputed lives + gradient boosting': 'Hybrid: physics-imputed labels + GBM',
         PH: 'Physics-informed degradation model', 'ORACLE gradient boosting (true failure labels)': 'Oracle: GBM with true labels*',
         'ORACLE physics model (fit on full lives)': 'Oracle: physics on full lives*'}
# ---- Fig 1: RMSE comparison
M['lo'] = M.RMSE_CI.str.split('-').str[0].astype(float); M['hi'] = M.RMSE_CI.str.split('-').str[1].astype(float)
M['oracle'] = M.model.str.startswith('ORACLE'); M = M.sort_values(['oracle', 'RMSE'], ascending=[True, False])
fig, ax = plt.subplots(figsize=(8.2, 4.2)); y = np.arange(len(M))
col = [BLUE if m == PH else (LGRAY if o else GRAY) for m, o in zip(M.model, M.oracle)]
ax.barh(y, M.RMSE.clip(upper=100), color=col, height=0.6)
ax.errorbar(M.RMSE.clip(upper=100), y, xerr=[M.RMSE.clip(upper=100) - M.lo.clip(upper=100), M.hi.clip(upper=100) - M.RMSE.clip(upper=100)],
            fmt='none', ecolor=INK2, lw=1, capsize=3)
for yi, v in zip(y, M.RMSE): ax.text(min(v, 100) + 1.5, yi + 0.28, f'{v:.1f} s', color=INK, fontsize=9)
ax.set_yticks(y, [short[m] for m in M.model]); ax.set_xlabel('RUL error on 50 test filters, RMSE (s); whiskers = 95% bootstrap CI')
ax.set_xlim(0, 112); ax.grid(axis='y', visible=False)
ax.set_title('Physics-informed model cuts RUL error 58% vs naive gradient boosting', loc='left', color=INK, fontsize=11)
fig.text(0.01, 0.01, '*Oracles use the uncensored training lives a real operator would not have; shown as a ceiling.', color=INK2, fontsize=8)
plt.tight_layout(rect=(0, 0.04, 1, 1)); plt.savefig('figures/fig1_model_comparison.png', dpi=160); plt.close()
# ---- Fig 2: predicted vs true
fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
for ax, name, c in [(axs[0], GB, ORANGE), (axs[1], PH, BLUE)]:
    ax.plot([0, 160], [0, 160], color=GRAY, lw=1, ls='--')
    for d, mk in [('A2', 'o'), ('A3', 's'), ('A4', '^')]:
        s = P[P.dust == d]; ax.scatter(s.true, s[name], s=34, marker=mk, color=c, edgecolor='#fcfcfb', lw=1, label=f'{d} dust')
    rm = np.sqrt(((P[name] - P.true) ** 2).mean())
    ax.set_title(f'{short[name]}  (RMSE {rm:.1f} s)', loc='left', color=INK, fontsize=10); ax.set_xlabel('True RUL (s)')
    ax.set_xlim(0, 160); ax.set_ylim(0, 175)
axs[0].set_ylabel('Predicted RUL (s)'); axs[1].legend(frameon=False, loc='lower right')
axs[1].text(30, 155, 'above line: predicts too late (risky)', color=INK2, fontsize=8); axs[1].text(95, 45, 'below line: early warning\n(safe, wastes some life)', color=INK2, fontsize=8)
plt.tight_layout(); plt.savefig('figures/fig2_pred_vs_true.png', dpi=160); plt.close()
# ---- Fig 3: worked example
tr, te = load('Train_data.csv'), load('Test_data.csv'); pm = PhysicsModel(sd_logk=0.06).fit(tr)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, n in zip(axs, [38, 30]):
    r = te[n]; tn = r['t'][-1]; z, _ = pm.unit_posterior(r, tn); K = pm.K(r['dust'], r['Q'], z[1]); p0 = pm.p0[r['dust']]
    mf = pm.mass_at_failure(r['dust'], r['Q'], *z); tf = mf / r['mrate']
    tt = np.linspace(0, max(tf, tn + r['rul_true']) * 1.05, 400)
    ax.plot(r['t'], r['dp'], color=INK2, lw=1, label='Measured ΔP (history available at prediction time)')
    ax.plot(tt, curve(tt * r['mrate'], z[0], K, p0), color=BLUE, lw=2, label='Physics model fitted to this filter')
    ax.axhline(600, color=GRAY, lw=1, ls='--'); ax.axvline(tn, color=LGRAY, lw=1)
    ax.plot([tn + r['rul_true']], [600], 'o', ms=8, color=INK, label='True failure'); ax.plot([tf], [600], 'o', ms=8, mfc='#fcfcfb', mec=BLUE, mew=2, label='Predicted failure')
    ax.set_ylim(0, 700); ax.set_xlabel('Time (s)'); ax.set_ylabel('Differential pressure (Pa)')
    ax.set_title(f'Test filter {n} ({r["dust"]} dust): predicted RUL {tf - tn:.0f} s, true {r["rul_true"]:.0f} s', loc='left', color=INK, fontsize=10)
axs[0].legend(frameon=False, fontsize=8, loc='upper left')
plt.tight_layout(); plt.savefig('figures/fig3_worked_example.png', dpi=160); plt.close()
# ---- Fig 4: LOO secondary
S = pd.read_csv('results/secondary_loo.csv'); g = S.groupby('frac').apply(lambda d: pd.Series({k: np.sqrt(((d[k] - d.true) ** 2).mean()) for k in ['gbm_naive', 'physics']}))
fig, ax = plt.subplots(figsize=(6.5, 3.8)); x = np.arange(3); w = 0.36
ax.bar(x - w / 2 - 0.01, g.gbm_naive, w, color=ORANGE, label='Gradient boosting, naive labels'); ax.bar(x + w / 2 + 0.01, g.physics, w, color=BLUE, label='Physics-informed model')
for xi, a, b in zip(x, g.gbm_naive, g.physics): ax.text(xi - w / 2, a + 1, f'{a:.0f}', ha='center', fontsize=9, color=INK); ax.text(xi + w / 2, b + 1, f'{b:.0f}', ha='center', fontsize=9, color=INK)
ax.set_xticks(x, ['25% of life', '50% of life', '75% of life']); ax.set_ylabel('RMSE (s)'); ax.grid(axis='x', visible=False)
ax.set_title('Leave-one-filter-out check on 50 more filters', loc='left', color=INK, fontsize=11); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig('figures/fig4_loo_by_stage.png', dpi=160); plt.close()
print('done')
