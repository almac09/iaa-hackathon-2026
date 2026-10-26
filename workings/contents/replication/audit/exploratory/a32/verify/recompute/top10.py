import sys, runpy
import numpy as np
ns = runpy.run_path('<scratch>/verify32/recompute/score2_lib.py')
evaluate, gain, y, line, R, allm = ns['evaluate'], ns['gain'], ns['y'], ns['line'], ns['R'], ns['allm']
ys = R.y_small.to_numpy(float)
def show(nm, msk, **kw):
    d, _ = evaluate(msk, **kw); print(nm, kw.keys() if kw else '', round(gain(d,'A2').mean(), 10), round(gain(d,'A3').mean(), 10))
top = np.argsort(-y, kind='stable')[:10]; km = allm.copy(); km[top] = False
show('pooled top10 resolved', km)
show('pooled top10 lev resolved base full', km, base_mask=allm)
show('pooled top10 lev full base resolved', km, lev_mask=allm)
show('pooled top10 per-line base', km, pooled='per_line')
for k in [9, 11, 12]:
    t = np.argsort(-y, kind='stable')[:k]; m = allm.copy(); m[t] = False; show(f'pooled top{k} resolved', m)
# by y_small
t = np.argsort(-ys, kind='stable')[:10]; m = allm.copy(); m[t] = False; show('top10 by y_small', m)
# by customer total
import pandas as pd
ct = R.groupby('customer_id').y_full.transform('sum').to_numpy()
t = np.argsort(-ct, kind='stable')[:10]; m = allm.copy(); m[t] = False; show('top10 rows by customer total', m)
print('top10 y', np.sort(y)[::-1][:12].round(1))
print('n_claims of top10', R.n_claims.to_numpy()[top])
