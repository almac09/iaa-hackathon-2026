import pandas as pd, numpy as np, json
D='<scratch>/a32/'
for line in ['contents','motor']:
    y24=pd.read_parquet(D+f'y24_{line}.parquet'); out=pd.read_parquet(D+f'y24out_{line}.parquet')
    pl=pd.read_parquet(D+f'pred_lgbm_{line}.parquet'); pt=pd.read_parquet(D+f'pred_tabpfn_{line}.parquet')
    print('==',line)
    print(' row_id unique', y24.row_id.is_unique, 'same order out/pl/pt', (y24.row_id.values==out.row_id.values).all(), (y24.row_id.values==pl.row_id.values).all(), (y24.row_id.values==pt.row_id.values).all())
    print(' same set', set(y24.row_id)==set(out.row_id)==set(pl.row_id)==set(pt.row_id))
    print(' cust match', (y24.customer_id.values==out.customer_id.values).all(), 'yr', y24.yr.unique())
    print(' NB out', out.new_business.astype(str).value_counts().to_dict(), (y24.new_business.astype(str).values==out.new_business.astype(str).values).mean() if 'new_business' in y24 else 'n/a')
    agecol = 'driver_age_band' if line=='motor' else 'policyholder_age_band'
    print(' age', y24[agecol].astype(str).value_counts().to_dict(), out.age_band.astype(str).value_counts().to_dict())
    print(' age match', (y24[agecol].astype(str).values==out.age_band.astype(str).values).mean())

    print(' expo', y24.exposure.describe().round(4).to_dict()); print(' expo_plain', y24.exposure_plain.describe().round(4).to_dict())
    print(' ratio expo/plain', (y24.exposure/y24.exposure_plain).describe().round(5).to_dict())
    print(' y_small', out.y_small.describe().round(2).to_dict(), 'y_full', out.y_full.describe().round(2).to_dict())
    print(' y_full>0 frac', (out.y_full>0).mean(), 'n_claims>0', (out.n_claims>0).mean(), 'consistent', ((out.y_full>0)==(out.n_claims>0)).mean())
    print(' ysmall<=yfull', (out.y_small<=out.y_full+1e-9).all(), 'small==full frac', (np.isclose(out.y_small,out.y_full)).mean())
    L=y24.c0_total-y24.c0_small
    print(' c0', y24.c0_small.describe().round(3).to_dict(), 'large min', L.min(), 'c0_total sum', y24.c0_total.sum(), 'y_full sum', out.y_full.sum())
    print(' nan', y24[['exposure','exposure_plain','c0_small','c0_total']].isna().sum().sum(), pl.isna().sum().sum(), pt.isna().sum().sum(), out.isna().sum().to_dict())
    print(' lgbm_source-L min', (pl.lgbm_source-L).min(), 'A2 min', pl.filter(like='A2_').min().min())
    for s in range(2032,2037):
        a1=pt[f'A3_w1.0_draw{s}']; a5=pt[f'A3_w0.5_draw{s}']
        r=(a1-L)/y24.c0_small; r5=(a5-L)/y24.c0_small
        print(f'  A3 {s} sqrt check maxabs', np.nanmax(np.abs(r5-np.sqrt(r))), 'r q', np.quantile(r,[0.01,0.5,0.99]).round(3), 'min A3', pt.filter(like=f'draw{s}').min().min())
    print(' sum ratios A2/c0', [round(pl[f'A2_draw{s}'].sum()/y24.c0_total.sum(),4) for s in range(2032,2037)], 'A3', [round(pt[f'A3_w0.5_draw{s}'].sum()/y24.c0_total.sum(),4) for s in range(2032,2037)])
