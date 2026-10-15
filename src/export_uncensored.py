"""Convert the MATLAB table in Train_Data_Uncensored.mat to CSV (scipy cannot read MATLAB tables directly)."""
import numpy as np, pandas as pd
from matdecode import subsystem_vars
v = subsystem_vars('Data/Train_Data_Uncensored.mat')[0][0, 0]
L = list(v.MCOS[0]['_ObjectMetadata'].flat)
data = [c for c in L if isinstance(c, np.ndarray) and c.dtype == object and c.shape == (1, 3)
        and isinstance(c[0, 0], np.ndarray) and c[0, 0].dtype.kind == 'f']
outer = next(c[0] for c in L if isinstance(c, np.ndarray) and c.dtype == object and c.shape == (1, 5) and isinstance(c[0, 1], np.ndarray) and c[0, 1].dtype.kind == 'f')
dust = [str(x[0][0][0]) for x in [d[0] for d in outer[2]]]; feed = outer[3].ravel(); no = outer[4].ravel()
rows = [pd.DataFrame(dict(Data_No=int(no[k]), Differential_pressure=np.ravel(c[0, 0]), Flow_rate=np.ravel(c[0, 1]),
        Time=np.ravel(c[0, 2]), Dust_feed=feed[k], Dust=dust[k])) for k, c in enumerate(data)]
pd.concat(rows).to_csv('Data/Train_data_uncensored.csv', index=False)
