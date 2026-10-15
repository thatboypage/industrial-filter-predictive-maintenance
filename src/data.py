"""Data loading for the Hagmeyer et al. (2021) 'Preventive to Predictive Maintenance' filter dataset."""
import numpy as np, pandas as pd, os
DATA = os.path.join(os.path.dirname(__file__), '..', 'Data')
RHO = {'A2': 0.900, 'A3': 1.025, 'A4': 1.200}   # bulk density g/cm3 from dataset documentation
FAIL_DP = 600.0                                  # Pa, failure threshold

def load(name):
    D = pd.read_csv(os.path.join(DATA, name))
    D['dust'] = D.Dust.str.extract(r'(A\d)')[0]
    runs = {}
    for n, g in D.groupby('Data_No'):
        g = g[g.Time.diff().fillna(1) > 0] if g.Time.iloc[0] > g.Time.iloc[1] else g   # drop glitch row (run 28)
        g = g[g.Time <= g.Time.iloc[-1]].reset_index(drop=True)
        g = g.iloc[np.argsort(g.Time.values, kind='stable')].reset_index(drop=True)
        d = g.dust.iloc[0]
        r = dict(no=int(n), dust=d, feed=float(g.Dust_feed.iloc[0]),
                 t=g.Time.values, dp=g.Differential_pressure.values, q=g.Flow_rate.values)
        r['mrate'] = r['feed'] * RHO[d] / 1000.0          # g/s of dust fed
        r['m'] = r['mrate'] * r['t']                      # cumulative dust mass fed (g)
        r['Q'] = float(np.median(r['q'][len(r['q'])//5:]))  # set-point flow (flow-controlled tests)
        if 'RUL' in g:
            r['rul_true'] = float(g.RUL.iloc[-1])
        runs[int(n)] = r
    return runs

def eol_time(r):
    i = np.where(r['dp'] >= FAIL_DP)[0]
    return float(r['t'][i[0]]) if len(i) else np.nan
