"""Physics-informed degradation model for filter clogging.

Depth-to-cake filtration transition modelled as a softplus in deposited dust mass m (g):
    dP(m) = p0 * softplus( K (m - m0) / p0 ),     K = k_i * c_dust * (Q/60)^beta
- For m << m0 (depth filtration) dP ~ 0; for m >> m0 (cake filtration) dP grows linearly with slope K,
  as in classical cake-filtration theory (dP proportional to deposited mass per area).
- Population parameters (c_dust, p0_dust, beta) are learned from training trajectories. Censoring does not
  bias this, because we model the measured degradation path, not the time-to-failure labels.
- Unit parameters (m0_i, k_i) are inferred for each new filter from its own history (MAP with a prior on k_i).
"""
import numpy as np
from scipy.optimize import least_squares
DUSTS = ['A2', 'A3', 'A4']
FAIL = 600.0

def softplus(z):
    return np.where(z > 30, z, np.log1p(np.exp(np.minimum(z, 30))))

def inv_softplus(y):
    return np.where(y > 30, y, np.log(np.expm1(np.maximum(y, 1e-12))))

def curve(m, m0, K, p0):
    return p0 * softplus(K * (m - m0) / p0)

def sigma(dp):                     # heteroscedastic measurement noise model (Pa)
    return 5.0 + 0.08 * dp

def downsample(r, t_max=None, step=10):
    sel = np.arange(0, len(r['t']), step)
    if t_max is not None:
        sel = sel[r['t'][sel] <= t_max]
    return r['m'][sel], r['dp'][sel]

class PhysicsModel:
    def __init__(self, sd_logk=0.15, m0_prior=True):
        self.sd_logk = sd_logk; self.m0_prior = m0_prior

    # ---------- population fit ----------
    def _unpack(self, th, n):
        c = dict(zip(DUSTS, np.exp(th[0:3]))); p0 = dict(zip(DUSTS, np.exp(th[3:6]))); beta = th[6]
        m0 = th[7:7 + n]; logk = th[7 + n:7 + 2 * n]
        return c, p0, beta, m0, logk

    def K(self, dust, Q, logk=0.0):
        return np.exp(logk) * self.c[dust] * (Q / 60.0) ** self.beta

    def fit(self, runs):
        runs = list(runs.values()); n = len(runs)
        data = [downsample(r) for r in runs]
        th0 = np.r_[np.log([30, 30, 30]), np.log([60, 60, 60]), 1.0,
                    [np.interp(100, d[1], d[0]) if d[1].max() > 100 else d[0].max() for d in data], np.zeros(n)]
        def res(th):
            c, p0, beta, m0, logk = self._unpack(th, n); out = []
            for i, (r, (m, dp)) in enumerate(zip(runs, data)):
                K = np.exp(logk[i]) * c[r['dust']] * (r['Q'] / 60.0) ** beta
                out.append((curve(m, m0[i], K, p0[r['dust']]) - dp) / sigma(dp))
            out.append(logk / self.sd_logk)          # random-effect prior
            return np.concatenate(out)
        sol = least_squares(res, th0, method='trf', x_scale='jac', max_nfev=3000)
        self.c, self.p0, self.beta, m0, logk = self._unpack(sol.x, n)
        self.train_logk = logk; self.train_m0 = m0
        self.sd_logk_emp = float(np.std(logk))
        dust = np.array([r['dust'] for r in runs])
        self.m0_mu = {d: float(np.mean(m0[dust == d])) for d in DUSTS}
        self.m0_sd = {d: float(np.std(m0[dust == d]) + 0.5) for d in DUSTS}
        return self

    # ---------- per-unit inference and RUL ----------
    def unit_posterior(self, r, t_now):
        m, dp = downsample(r, t_now, step=5)
        d, Q, p0 = r['dust'], r['Q'], self.p0[r['dust']]
        m0_init = np.interp(min(dp.max(), 100), np.maximum.accumulate(dp), m) if dp.max() > 20 else max(m[-1] + 1, self.m0_mu[d])
        def res(z):
            K = self.K(d, Q, z[1])
            pri = [(z[0] - self.m0_mu[d]) / self.m0_sd[d]] if self.m0_prior else []
            return np.r_[(curve(m, z[0], K, p0) - dp) / sigma(dp), z[1] / self.sd_logk, pri]
        sol = least_squares(res, [m0_init, 0.0], method='trf')
        J = sol.jac; cov = np.linalg.pinv(J.T @ J)
        return sol.x, cov

    def mass_at_failure(self, d, Q, m0, logk):
        p0 = self.p0[d]; K = self.K(d, Q, logk)
        return m0 + p0 / K * inv_softplus(FAIL / p0)

    def predict_rul(self, r, t_now, n_samples=400, rng=None):
        rng = rng or np.random.default_rng(0)
        z, cov = self.unit_posterior(r, t_now)
        m_now = r['mrate'] * t_now
        mf = self.mass_at_failure(r['dust'], r['Q'], *z)
        rul = max(mf - m_now, 0.0) / r['mrate']
        S = rng.multivariate_normal(z, cov + 1e-9 * np.eye(2), n_samples)
        mfs = np.array([self.mass_at_failure(r['dust'], r['Q'], *s) for s in S])
        ruls = np.maximum(mfs - m_now, 0.0) / r['mrate']
        return rul, np.percentile(ruls, [5, 50, 95])
