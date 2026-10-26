# Databricks notebook source
# DBTITLE 1,Developed Market v5 External LDFs
# MAGIC %md
# MAGIC # Developed Market v5 — External LDFs
# MAGIC
# MAGIC Applies external loss development factors (BI, subsidence) to the **developed market** proc files,
# MAGIC then re-trains XGBoost + GLM on the corrected `ultimate_claims` targets.
# MAGIC
# MAGIC **Key change from v4:** BI and subsidence targets scaled by external LDFs from theactuarialclub.com
# MAGIC (replaces Part 1 count-only factors). Short-tail perils unchanged.
# MAGIC
# MAGIC **v4 baseline D2:** XGBoost 0.0553, GLM 0.077, Blend 0.0844

# COMMAND ----------

# DBTITLE 1,Step 1: Load data + apply external LDFs + update proc files
# Step 1: Load data, apply external LDFs, update proc files
import pandas as pd
import numpy as np
import os

%pip install xgboost statsmodels openpyxl -q

BASE = '/Workspace/Users/bridget.macdonnell1@gmail.com/iaa-hackathon-2026'
DATA_DIR = f'{BASE}/data/'
PROC_DIR = os.path.join(DATA_DIR, 'proc')
V5_DIR = f'{BASE}/files_v5/'
os.makedirs(V5_DIR, exist_ok=True)

# ============================================================
# EXTERNAL LDFs (same as 04 notebook)
# ============================================================
RAW_LDFS = {
    'bodily_injury': {12: (1.85, 2.40), 24: (1.40, 1.65), 36: (1.20, 1.35), 48: (1.10, 1.18), 60: (1.05, 1.08), 'tail': (1.02, 1.05)},
    'subsidence': {12: (1.50, 1.80), 24: (1.15, 1.25), 36: (1.05, 1.10), 48: (1.01, 1.03), 60: (1.00, 1.00), 'tail': (1.00, 1.00)},
}
LDF_MID = {p: {a: (r[0]+r[1])/2 for a, r in f.items()} for p, f in RAW_LDFS.items()}

SNAPSHOT = pd.Timestamp('2024-12-31')
loss_years = [2021, 2022, 2023, 2024]
dev_ages = {yr: (SNAPSHOT - pd.Timestamp(f'{yr}-07-01')).days / 30.44 for yr in loss_years}

def interp_ldf(peril, age):
    t = LDF_MID[peril]; ages = [12, 24, 36, 48, 60]
    if age <= 12: return t[12]
    elif age >= 60: return t[60] * t['tail']
    for i in range(len(ages)-1):
        if ages[i] <= age <= ages[i+1]:
            frac = (age - ages[i]) / (ages[i+1] - ages[i])
            return t[ages[i]] + frac * (t[ages[i+1]] - t[ages[i]])
    return t[60]

YEAR_LDFS = {yr: {'bodily_injury': interp_ldf('bodily_injury', dev_ages[yr]),
                  'subsidence': interp_ldf('subsidence', dev_ages[yr])} for yr in loss_years}
print('=== EXTERNAL LDFs BY LOSS YEAR ===')
for yr in loss_years:
    print(f'  {yr}: BI={YEAR_LDFS[yr]["bodily_injury"]:.3f}, Subs={YEAR_LDFS[yr]["subsidence"]:.3f}')

# Part 1 count-only factors (for short-tail perils and for computing scaling ratio)
P1_FACTORS = {
    2021: {'bi': 1.00, 'tpp': 1.00, 'short_motor': 1.00, 'subsidence': 1.00, 'buildings_other': 1.00, 'contents': 1.00},
    2022: {'bi': 1.01, 'tpp': 1.01, 'short_motor': 1.01, 'subsidence': 1.05, 'buildings_other': 1.02, 'contents': 1.01},
    2023: {'bi': 1.06, 'tpp': 1.01, 'short_motor': 1.01, 'subsidence': 1.15, 'buildings_other': 1.02, 'contents': 1.01},
    2024: {'bi': 1.82, 'tpp': 1.08, 'short_motor': 1.08, 'subsidence': 1.47, 'buildings_other': 1.08, 'contents': 1.04},
}

def get_new_factor(peril, line, yr):
    p = str(peril).lower()
    if p == 'bodily_injury': return YEAR_LDFS.get(yr, {}).get('bodily_injury', 1.0)
    elif p == 'subsidence': return YEAR_LDFS.get(yr, {}).get('subsidence', 1.0)
    elif line == 'motor' and 'third_party' in p: return P1_FACTORS.get(yr, {}).get('tpp', 1.0)
    elif line == 'motor': return P1_FACTORS.get(yr, {}).get('short_motor', 1.0)
    elif line == 'buildings': return P1_FACTORS.get(yr, {}).get('buildings_other', 1.0)
    else: return P1_FACTORS.get(yr, {}).get('contents', 1.0)

def get_p1_factor(peril, line, yr):
    p = str(peril).lower()
    if p == 'bodily_injury': tail = 'bi'
    elif p == 'subsidence': tail = 'subsidence'
    elif line == 'motor' and 'third_party' in p: tail = 'tpp'
    elif line == 'motor': tail = 'short_motor'
    elif line == 'buildings': tail = 'buildings_other'
    else: tail = 'contents'
    return P1_FACTORS.get(yr, {}).get(tail, 1.0)

# ============================================================
# LOAD CLAIMS + COMPUTE SCALING
# ============================================================
claims = pd.read_parquet(f'{BASE}/data/claims.parquet')
claims['loss_year'] = pd.to_datetime(claims['loss_date']).dt.year
claims['incurred_net'] = claims['incurred_to_date'] - claims['recoveries']
claims = claims[claims['status'] != 'repudiated'].copy()

# Only use years 2021-2024 (matching proc files)
claims = claims[claims['loss_year'].isin([2021, 2022, 2023, 2024])].copy()

claims['p1_factor'] = claims.apply(lambda r: get_p1_factor(r['peril'], r['line'], r['loss_year']), axis=1)
claims['new_factor'] = claims.apply(lambda r: get_new_factor(r['peril'], r['line'], r['loss_year']), axis=1)
claims['factor_ratio'] = claims['new_factor'] / claims['p1_factor']

# Aggregate scaling by policy
policy_scaling = claims.groupby(['line', 'policy_id']).apply(
    lambda g: np.average(g['factor_ratio'], weights=g['incurred_net']) if g['incurred_net'].sum() > 0 else 1.0
).reset_index(name='scaling_factor')

print('\n=== POLICY-LEVEL SCALING ===')
for line in ['motor', 'buildings', 'contents']:
    ls = policy_scaling[policy_scaling['line'] == line]
    print(f'{line}: {len(ls)} policies with claims, mean scale={ls["scaling_factor"].mean():.4f}')

# ============================================================
# UPDATE PROC FILES
# ============================================================
train_v5 = {}
for line in ['motor', 'buildings', 'contents']:
    train_v5[line] = pd.read_parquet(os.path.join(PROC_DIR, f'proc_{line}.parquet'))
    old_mean = train_v5[line]['ultimate_claims'].mean()
    ls = policy_scaling[policy_scaling['line'] == line][['policy_id', 'scaling_factor']]
    train_v5[line] = train_v5[line].merge(ls, on='policy_id', how='left')
    train_v5[line]['scaling_factor'] = train_v5[line]['scaling_factor'].fillna(1.0)
    train_v5[line]['ultimate_claims'] = train_v5[line]['ultimate_claims'] * train_v5[line]['scaling_factor']
    train_v5[line].drop('scaling_factor', axis=1, inplace=True)
    new_mean = train_v5[line]['ultimate_claims'].mean()
    print(f'{line}: old mean={old_mean:.4f}, new mean={new_mean:.4f}, ratio={new_mean/old_mean:.4f}')
    # Save v5 proc file
    train_v5[line].to_parquet(os.path.join(V5_DIR, f'proc_{line}_v5.parquet'), index=False)

print('\nSaved v5 proc files to files_v5/')

# COMMAND ----------

# DBTITLE 1,Step 2: Test prep + XGBoost v5
# Step 2: Prepare test data + run XGBoost
import xgboost as xgb

# Load test files + enrichment
test = {}
for line in ['motor', 'buildings', 'contents']:
    test[line] = pd.read_parquet(os.path.join(DATA_DIR, f'test_{line}.parquet'))
customers = pd.read_parquet(os.path.join(DATA_DIR, 'customers.parquet'))
addresses = pd.read_parquet(os.path.join(DATA_DIR, 'addresses.parquet'))
weather = pd.read_parquet(os.path.join(DATA_DIR, 'weather.parquet'))
sample_submission = pd.read_parquet(os.path.join(DATA_DIR, 'sample_submission.parquet'))

# Clean test files
for line in ['motor', 'buildings', 'contents']:
    df = test[line]
    ps = pd.to_datetime(df['period_start'])
    pe = pd.to_datetime(df['period_end'])
    df['exposure'] = ((pe - ps).dt.days + 1) / 365.0
    df['Year'] = ps.dt.year
    for col in ['policy_id', 'customer_id', 'address_id', 'row_id']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if 'compulsory_excess' in df.columns:
        df.drop('compulsory_excess', axis=1, inplace=True)

test['buildings']['construction_type'] = test['buildings']['construction_type'].astype(str).str.strip().str.upper().replace({'W': 'TIMBER', 'T': 'TIMBER'})
test['motor']['customer_id'] = test['motor']['customer_id'].str.zfill(8)
customers['customer_id'] = customers['customer_id'].astype(str).str.zfill(8)

customers_clean = customers.drop(columns=['surname', 'date_of_birth'], errors='ignore')
weather_agg = weather.groupby(['address_id', 'year'], as_index=False).mean(numeric_only=True)
weather_agg.rename(columns={'year': 'Year'}, inplace=True)

for line in ['motor', 'buildings', 'contents']:
    df = test[line]
    df = df.merge(customers_clean, on='customer_id', how='left', suffixes=('', '_cust'))
    df = df.merge(addresses, on='address_id', how='left', suffixes=('', '_addr'))
    df = df.merge(weather_agg, on=['address_id', 'Year'], how='left', suffixes=('', '_weather'))
    test[line] = df
    print(f'test_{line}: {df.shape}')

# Load feature selection
FEATURES_PATH = f'{BASE}/notebooks_v4/Features_v3.xlsx'
sheet_map = {'motor': 'Motor', 'buildings': 'Building', 'contents': 'Contents'}
feature_selection = {}
for line, sheet in sheet_map.items():
    df = pd.read_excel(FEATURES_PATH, sheet_name=sheet)
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
    feature_selection[line] = df[df.iloc[:, 1] == 1].iloc[:, 0].tolist()
    print(f'{line}: {len(feature_selection[line])} features')

# Tweedie D2
def tweedie_deviance(y, y_pred, p=1.5):
    y = np.array(y, dtype=float)
    y_pred = np.maximum(y_pred, 1e-20)
    return 2 * (np.power(np.maximum(y, 0), 2-p) / ((1-p)*(2-p))
                - y * np.power(y_pred, 1-p) / (1-p)
                + np.power(y_pred, 2-p) / (2-p))

def tweedie_d2(y, y_pred, p=1.5):
    dev = tweedie_deviance(y, y_pred, p).sum()
    dev_null = tweedie_deviance(y, np.mean(y), p).sum()
    return 1 - dev / dev_null

# XGBoost params (same as v4)
XGB_PARAMS = {
    'objective': 'reg:tweedie', 'tweedie_variance_power': 1.5,
    'max_depth': 6, 'eta': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.5,
    'min_child_weight': 10, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
    'seed': 42, 'n_jobs': -1, 'verbosity': 0,
}

# New trend factors (from 04 notebook analysis)
NEW_TREND = {'motor': 1.048, 'buildings': 1.090, 'contents': 1.049}

xgb_models = {}
xgb_val_d2 = {}
xgb_val_preds = {}
xgb_test_preds = {}
xgb_importance = {}

for line in ['motor', 'buildings', 'contents']:
    print(f'\n{"="*60}')
    print(f'=== {line.upper()} XGBoost v5 ===')
    print(f'{"="*60}')
    
    train_df = train_v5[line]
    test_df = test[line]
    
    # Select features
    feats = [f for f in feature_selection[line] if f in train_df.columns and f not in ['ultimate_claims', 'row_id', 'policy_id', 'customer_id', 'address_id', 'period_start', 'period_end']]
    print(f'  Features: {len(feats)}')
    
    X_all = train_df[feats].copy()
    X_test = test_df[feats].copy()
    
    # Encode categoricals
    for col in feats:
        if X_all[col].dtype == 'object' or X_all[col].dtype == 'bool':
            cats = pd.Categorical(X_all[col])
            X_all[col] = cats.codes.astype('float64')
            X_test[col] = pd.Categorical(X_test[col], categories=cats.categories).codes.astype('float64')
    
    y_all = train_df['ultimate_claims'].values
    exp_all = train_df['exposure'].values
    year = train_df['Year'].values
    
    train_mask = year <= 2023
    val_mask = year == 2024
    
    dtrain = xgb.DMatrix(X_all[train_mask], label=y_all[train_mask], base_margin=np.log(exp_all[train_mask] + 1e-10), weight=exp_all[train_mask])
    dval = xgb.DMatrix(X_all[val_mask], label=y_all[val_mask], base_margin=np.log(exp_all[val_mask] + 1e-10), weight=exp_all[val_mask])
    dtest = xgb.DMatrix(X_test, base_margin=np.log(test_df['exposure'].values + 1e-10))
    
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=300, evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
    xgb_models[line] = model
    
    val_pred = np.maximum(model.predict(dval), 0)
    d2 = tweedie_d2(y_all[val_mask], val_pred)
    xgb_val_d2[line] = d2
    xgb_val_preds[line] = val_pred
    print(f'  Val D2: {d2:.4f} (v4 was: {0.0599 if line=="motor" else 0.0588 if line=="buildings" else 0.0470:.4f})')
    print(f'  Val pred mean: {val_pred.mean():.4f} (actual: {y_all[val_mask].mean():.4f})')
    
    # Feature importance
    scores = model.get_score(importance_type='gain')
    imp = pd.DataFrame({'feature': feats, 'importance': [scores.get(f, 0) for f in feats]}).sort_values('importance', ascending=False)
    xgb_importance[line] = imp
    print(f'  Top 5: {imp.head(5)["feature"].tolist()}')
    
    xgb_test_preds[line] = np.maximum(model.predict(dtest), 0.01)
    print(f'  Test pred mean: {xgb_test_preds[line].mean():.4f}')

print(f'\nXGBoost v5 avg D2: {np.mean(list(xgb_val_d2.values())):.4f} (v4 was 0.0553)')

# COMMAND ----------

# DBTITLE 1,Step 3: GLM v5
# Step 3: GLM v5 (Tweedie, top features from XGBoost importance)
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

SMOOTHING = 30
MAX_FEATS = 25

def target_encode(train_series, train_target, apply_series, smoothing=30):
    prior = train_target.mean()
    grp = pd.DataFrame({'cat': train_series, 'target': train_target})
    stats = grp.groupby('cat')['target'].agg(['mean', 'count'])
    smooth = (stats['mean'] * stats['count'] + prior * smoothing) / (stats['count'] + smoothing)
    return train_series.map(smooth).fillna(prior), apply_series.map(smooth).fillna(prior)

glm_val_d2 = {}
glm_val_preds = {}
glm_test_preds = {}

for line in ['motor', 'buildings', 'contents']:
    print(f'\n{"="*60}')
    print(f'=== {line.upper()} GLM v5 ===')
    print(f'{"="*60}')
    
    train_df = train_v5[line]
    test_df = test[line]
    
    # Select top features from XGBoost importance
    top_feats = xgb_importance[line]['feature'].head(MAX_FEATS).tolist()
    top_feats = [f for f in top_feats if f in train_df.columns and f not in ['ultimate_claims', 'row_id', 'policy_id', 'customer_id', 'address_id', 'period_start', 'period_end']]
    if 'Year' not in top_feats and 'Year' in train_df.columns:
        top_feats.append('Year')
    print(f'  Features: {len(top_feats)}')
    
    year = train_df['Year'].values
    train_mask = year <= 2023
    val_mask = year == 2024
    
    y_tr = train_df.loc[train_mask, 'ultimate_claims'].values
    y_val = train_df.loc[val_mask, 'ultimate_claims'].values
    exp_tr = train_df.loc[train_mask, 'exposure'].values
    exp_val = train_df.loc[val_mask, 'exposure'].values
    exp_test = test_df['exposure'].values
    
    # Prepare features
    X_tr = pd.DataFrame(index=train_df.index[train_mask])
    X_val = pd.DataFrame(index=train_df.index[val_mask])
    X_test_glm = pd.DataFrame(index=test_df.index)
    
    for col in top_feats:
        tr_vals = train_df.loc[train_mask, col]
        val_vals = train_df.loc[val_mask, col]
        test_vals = test_df[col]
        is_cat = tr_vals.dtype == 'object' or tr_vals.dtype == 'bool'
        if is_cat:
            tr_enc, val_enc = target_encode(tr_vals, y_tr, val_vals, SMOOTHING)
            _, test_enc = target_encode(tr_vals, y_tr, test_vals, SMOOTHING)
            X_tr[col] = tr_enc.values
            X_val[col] = val_enc.values
            X_test_glm[col] = test_enc.values
        else:
            med = tr_vals.median()
            X_tr[col] = tr_vals.fillna(med).values
            X_val[col] = val_vals.fillna(med).values
            X_test_glm[col] = test_vals.fillna(med).values
    
    # Standardize
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test_glm), columns=X_test_glm.columns, index=X_test_glm.index)
    
    X_tr_s = sm.add_constant(X_tr_s, has_constant='add')
    X_val_s = sm.add_constant(X_val_s, has_constant='add')
    X_test_s = sm.add_constant(X_test_s, has_constant='add')
    
    try:
        model = sm.GLM(y_tr, X_tr_s, family=sm.families.Tweedie(link=sm.families.links.Log(), var_power=1.5), offset=np.log(exp_tr + 1e-10))
        result = model.fit(max_iter=100)
        print(f'  Converged: {result.converged}')
        val_pred = np.maximum(result.predict(X_val_s, offset=np.log(exp_val + 1e-10)), 0)
        d2 = tweedie_d2(y_val, val_pred)
        glm_val_d2[line] = d2
        glm_val_preds[line] = val_pred
        print(f'  Val D2: {d2:.4f} (v4 was: {0.0577 if line=="motor" else 0.0905 if line=="buildings" else 0.0826:.4f})')
        print(f'  Val pred mean: {val_pred.mean():.4f} (actual: {y_val.mean():.4f})')
        glm_test_preds[line] = np.maximum(result.predict(X_test_s, offset=np.log(exp_test + 1e-10)), 0.01)
        print(f'  Test pred mean: {glm_test_preds[line].mean():.4f}')
    except Exception as e:
        print(f'  GLM FAILED: {e}')
        glm_val_d2[line] = -0.01
        glm_val_preds[line] = np.full(len(y_val), y_tr.mean())
        glm_test_preds[line] = np.full(len(test_df), y_tr.mean())

print(f'\nGLM v5 avg D2: {np.mean(list(glm_val_d2.values())):.4f} (v4 was 0.077)')

# COMMAND ----------

# DBTITLE 1,Step 4: Blend + submission
# Step 4: Blend + generate submissions
SUB_DIR = f'{BASE}/submissions_v5/'
os.makedirs(SUB_DIR, exist_ok=True)

print('=== BLEND v5 ===')
best_weights = {}
blend_d2 = {}

for line in ['motor', 'buildings', 'contents']:
    train_df = train_v5[line]
    year = train_df['Year'].values
    val_mask = year == 2024
    y_val = train_df.loc[val_mask, 'ultimate_claims'].values
    
    best_d2 = -999
    best_w = (0.5, 0.5)
    for wx in np.arange(0, 1.01, 0.05):
        wg = 1.0 - wx
        blend = wx * xgb_val_preds[line] + wg * glm_val_preds[line]
        d2 = tweedie_d2(y_val, blend)
        if d2 > best_d2:
            best_d2 = d2
            best_w = (wx, wg)
    blend_d2[line] = best_d2
    best_weights[line] = best_w
    print(f'{line}: XGB={best_w[0]:.2f}, GLM={best_w[1]:.2f} -> D2={best_d2:.4f} (v4: {0.0844:.4f})')

print(f'\nBlend v5 avg D2: {np.mean(list(blend_d2.values())):.4f} (v4 was 0.0844)')

# Generate calibrated submission
submission = sample_submission.copy()
for line in ['motor', 'buildings', 'contents']:
    test_df = test[line]
    wx, wg = best_weights[line]
    xgb_p = np.clip(xgb_test_preds[line], 0.01, 50000) * NEW_TREND[line]
    glm_p = np.clip(np.nan_to_num(glm_test_preds[line], posinf=50000, neginf=0.01), 0.01, 50000) * NEW_TREND[line]
    blend_pred = np.clip(wx * xgb_p + wg * glm_p, 0.01, 50000)
    # Calibrate to validation level
    cal_target = train_v5[line].loc[train_v5[line]['Year'] == 2024, 'ultimate_claims'].mean() * NEW_TREND[line]
    cal_factor = cal_target / max(blend_pred.mean(), 0.01)
    blend_pred = np.clip(blend_pred * cal_factor, 0.01, 50000)
    print(f'{line}: blend mean={blend_pred.mean():.4f} (target={cal_target:.4f})')
    pred_map = dict(zip(test_df['row_id'], blend_pred))
    mask = submission['row_id'].isin(pred_map)
    submission.loc[mask, 'prediction'] = submission.loc[mask, 'row_id'].map(pred_map)

submission['prediction'] = submission['prediction'].clip(lower=0.01, upper=50000)
submission['prediction'] = submission['prediction'].replace([np.inf, -np.inf], 50000).fillna(0.01)

sub_path = os.path.join(SUB_DIR, 'blend_v5_external_LDFs.csv')
submission.to_csv(sub_path, index=False)
print(f'\nSaved: {sub_path} ({len(submission)} rows, mean={submission["prediction"].mean():.4f})')
print(f'v4 submission mean was: 339.10')

# Also save XGBoost-only and GLM-only submissions
for line in ['motor', 'buildings', 'contents']:
    test_df = test[line]
    # XGBoost only
    xgb_p = np.clip(xgb_test_preds[line], 0.01, 50000) * NEW_TREND[line]
    pred_map = dict(zip(test_df['row_id'], xgb_p))
    mask = submission['row_id'].isin(pred_map)
    sub_xgb = sample_submission.copy()
    sub_xgb.loc[mask, 'prediction'] = sub_xgb.loc[mask, 'row_id'].map(pred_map)
    sub_xgb['prediction'] = sub_xgb['prediction'].clip(lower=0.01, upper=50000).fillna(0.01)

# Save summary
import pickle
v5_results = {
    'xgb_val_d2': xgb_val_d2, 'glm_val_d2': glm_val_d2, 'blend_d2': blend_d2,
    'best_weights': best_weights, 'trend_factors': NEW_TREND,
}
with open(os.path.join(V5_DIR, 'v5_results.pkl'), 'wb') as f:
    pickle.dump(v5_results, f)
print(f'Saved: {V5_DIR}v5_results.pkl')

# COMMAND ----------

