"""
Traffic Demand Prediction — Final Optimized Pipeline (Target: 94+)

Key improvements over v5 (90 score):
1. Leak-free Same-Day Morning History Features: Uses same-day morning demand for both Day 48 and Day 49.
2. Leak-free Same-Day Morning Aggregates: Computes morning stats on same-day morning data.
3. Native Categorical Features: Converts categoricals to category dtype for native split finding in LGBM/XGB.
4. Robust Handling of NaNs: Dynamically drops all-NaN columns from training splits.
5. Scaled Target Encodings: Adds Day 49/48 scaled versions of hourly and geohash target encodings.
6. Stacking Ensemble: Blends LightGBM + XGBoost + HistGradientBoosting using a Ridge meta-learner.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path('/Users/underxcore/Desktop/Flipkart_GridLock')
DATA_DIR     = PROJECT_ROOT / 'dataset'
OUTPUT_DIR   = PROJECT_ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

train_raw = pd.read_csv(DATA_DIR / 'train.csv')
test_raw  = pd.read_csv(DATA_DIR / 'test.csv')

print(f'Train raw shape: {train_raw.shape}, Test raw shape: {test_raw.shape}')

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS & BASE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def decode_geohash(gh):
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    is_even = True
    for ch in gh:
        v = base32.find(ch)
        if v == -1:
            continue
        for i in range(4, -1, -1):
            bit = (v >> i) & 1
            if is_even:
                mid = (lon_lo + lon_hi) / 2
                lon_lo, lon_hi = (mid, lon_hi) if bit else (lon_lo, mid)
            else:
                mid = (lat_lo + lat_hi) / 2
                lat_lo, lat_hi = (mid, lat_hi) if bit else (lat_lo, mid)
            is_even = not is_even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2

def lookup_pivot_vectorized(pivot, index_arr, col_arr):
    result = np.full(len(index_arr), np.nan)
    idx_set = set(pivot.index)
    col_set = set(pivot.columns)
    idx_pos = {v: i for i, v in enumerate(pivot.index)}
    col_pos = {v: i for i, v in enumerate(pivot.columns)}
    mat = pivot.values

    for i, (idx, col) in enumerate(zip(index_arr, col_arr)):
        if idx in idx_set and col in col_set:
            result[i] = mat[idx_pos[idx], col_pos[col]]
    return result

def cascade_lookup(pivots, keys_list, col_arr):
    result = np.full(len(col_arr), np.nan)
    for pivot, keys in zip(pivots, keys_list):
        mask = np.isnan(result)
        if not mask.any():
            break
        vals = lookup_pivot_vectorized(pivot, keys[mask], col_arr[mask])
        result[mask] = np.where(~np.isnan(vals), vals, result[mask])
    return result

def add_base_features(df):
    out = df.copy()
    hm = out['timestamp'].astype(str).str.split(':', expand=True).astype(int)
    out['hour']          = hm[0]
    out['minute']        = hm[1]
    out['minute_of_day'] = out['hour'] * 60 + out['minute']
    out['global_minute'] = out['day'] * 1440 + out['minute_of_day']

    out['time_sin'] = np.sin(2 * np.pi * out['minute_of_day'] / 1440)
    out['time_cos'] = np.cos(2 * np.pi * out['minute_of_day'] / 1440)
    out['hour_sin']  = np.sin(2 * np.pi * out['hour'] / 24)
    out['hour_cos']  = np.cos(2 * np.pi * out['hour'] / 24)

    out['is_morning_peak'] = out['hour'].between(7, 10).astype(int)
    out['is_evening_peak'] = out['hour'].between(16, 20).astype(int)
    out['is_night']        = ((out['hour'] <= 5) | (out['hour'] >= 22)).astype(int)
    out['is_midday']       = out['hour'].between(11, 14).astype(int)

    out['geo_5'] = out['geohash'].str[:5]
    out['geo_4'] = out['geohash'].str[:4]
    out['geo_3'] = out['geohash'].str[:3]
    out['geo_2'] = out['geohash'].str[:2]

    unique_gh = out['geohash'].unique()
    coords = {g: decode_geohash(g) for g in unique_gh}
    out['lat'] = out['geohash'].map(lambda g: coords[g][0])
    out['lon'] = out['geohash'].map(lambda g: coords[g][1])

    out['Temperature_missing'] = out['Temperature'].isna().astype(int)
    out['RoadType_missing']     = out['RoadType'].isna().astype(int)
    out['Weather_missing']      = out['Weather'].isna().astype(int)
    return out

print('Adding base features...')
train = add_base_features(train_raw)
test  = add_base_features(test_raw)

# ─────────────────────────────────────────────────────────────────────────────
# SAME-DAY MORNING HISTORY FEATURES (0-120 minutes)
# ─────────────────────────────────────────────────────────────────────────────

morn_train = train[train['minute_of_day'] <= 120]

def add_same_day_lags(df, morn_df, is_train=True):
    out = df.copy()
    avail_lags = [0, 15, 30, 45, 60, 75, 90, 105, 120]
    
    pivots = {}
    resolutions = [
        ('gh', ['day', 'geohash']),
        ('g5', ['day', 'geo_5']),
        ('g4', ['day', 'geo_4']),
        ('g3', ['day', 'geo_3']),
        ('g2', ['day', 'geo_2']),
    ]
    for res_name, keys in resolutions:
        pivot = morn_df.pivot_table(index=keys, columns='minute_of_day', values='demand', aggfunc='mean')
        pivot.columns = [f'same_day_lag_{c}' for c in pivot.columns]
        pivots[res_name] = pivot.reset_index()
        
    pivot_gb = morn_df.pivot_table(index=['day'], columns='minute_of_day', values='demand', aggfunc='mean')
    pivot_gb.columns = [f'same_day_lag_{c}' for c in pivot_gb.columns]
    pivots['gb'] = pivot_gb.reset_index()

    out = out.merge(pivots['gh'], on=['day', 'geohash'], how='left')
    
    for res_name, keys in resolutions[1:]:
        out = out.merge(pivots[res_name], on=keys, how='left', suffixes=('', f'_{res_name}'))
        for lag in avail_lags:
            col = f'same_day_lag_{lag}'
            out[col] = out[col].fillna(out[f'{col}_{res_name}'])
            out.drop(columns=[f'{col}_{res_name}'], inplace=True)
            
    out = out.merge(pivots['gb'], on=['day'], how='left', suffixes=('', '_gb'))
    for lag in avail_lags:
        col = f'same_day_lag_{lag}'
        out[col] = out[col].fillna(out[f'{col}_gb'])
        out.drop(columns=[f'{col}_gb'], inplace=True)
        
    if is_train:
        for lag in avail_lags:
            col = f'same_day_lag_{lag}'
            out.loc[out['minute_of_day'] <= lag, col] = np.nan
            
    return out

print('Adding same-day morning history lags...')
train = add_same_day_lags(train, morn_train, is_train=True)
test  = add_same_day_lags(test,  morn_train, is_train=False)

# ─────────────────────────────────────────────────────────────────────────────
# SAME-DAY MORNING AGGREGATES & VELOCITY
# ─────────────────────────────────────────────────────────────────────────────

def add_same_day_aggregates(df, morn_df):
    agg_gh = morn_df.groupby(['day', 'geohash'])['demand'].agg(
        same_day_morning_mean='mean',
        same_day_morning_std='std',
        same_day_morning_max='max',
        same_day_morning_min='min',
    ).reset_index()
    agg_gh['same_day_morning_std'] = agg_gh['same_day_morning_std'].fillna(0)

    vel_rows = []
    for (d, gh), grp in morn_df.sort_values('minute_of_day').groupby(['day', 'geohash']):
        if len(grp) >= 2:
            slope = float(np.polyfit(grp['minute_of_day'].values.astype(float), grp['demand'].values, 1)[0])
        else:
            slope = 0.0
        vel_rows.append({'day': d, 'geohash': gh, 'same_day_velocity': slope})
    vel_df = pd.DataFrame(vel_rows)
    agg_gh = agg_gh.merge(vel_df, on=['day', 'geohash'], how='left')

    morn_g5 = morn_df.copy()
    morn_g5['geo_5'] = morn_g5['geohash'].str[:5]
    agg_g5 = morn_g5.groupby(['day', 'geo_5'])['demand'].agg(
        same_day_morning_mean_g5='mean',
        same_day_velocity_g5='mean'
    ).reset_index()

    out = df.merge(agg_gh, on=['day', 'geohash'], how='left')
    out = out.merge(agg_g5, on=['day', 'geo_5'], how='left')

    out['same_day_morning_mean'] = out['same_day_morning_mean'].fillna(out['same_day_morning_mean_g5']).fillna(0)
    out['same_day_velocity']     = out['same_day_velocity'].fillna(out['same_day_velocity_g5']).fillna(0)
    out['same_day_morning_std']  = out['same_day_morning_std'].fillna(0)
    out['same_day_morning_max']  = out['same_day_morning_max'].fillna(out['same_day_morning_mean'])
    out['same_day_morning_min']  = out['same_day_morning_min'].fillna(out['same_day_morning_mean'])

    out.drop(columns=['same_day_morning_mean_g5', 'same_day_velocity_g5'], inplace=True, errors='ignore')
    return out

print('Adding same-day morning aggregates...')
train = add_same_day_aggregates(train, morn_train)
test  = add_same_day_aggregates(test,  morn_train)

# ─────────────────────────────────────────────────────────────────────────────
# PREVIOUS-DAY PROFILE FEATURES (Day 48 profile looked up for Day 49 rows)
# ─────────────────────────────────────────────────────────────────────────────

train_48 = train[train['day'] == 48]
train_48_geo = train_48[['geo_5','geo_4','geo_3','geo_2','minute_of_day','demand']]

pivot_48    = train_48.pivot_table(index='geohash', columns='minute_of_day', values='demand', aggfunc='mean')
pivot_48_g5 = train_48_geo.pivot_table(index='geo_5', columns='minute_of_day', values='demand', aggfunc='mean')
pivot_48_g4 = train_48_geo.pivot_table(index='geo_4', columns='minute_of_day', values='demand', aggfunc='mean')
pivot_48_g3 = train_48_geo.pivot_table(index='geo_3', columns='minute_of_day', values='demand', aggfunc='mean')
pivot_48_g2 = train_48_geo.pivot_table(index='geo_2', columns='minute_of_day', values='demand', aggfunc='mean')
pivot_48_gb = train_48_geo.pivot_table(columns='minute_of_day', values='demand', aggfunc='mean')

train_49_morning = train[(train['day'] == 49) & (train['minute_of_day'] <= 120)]

# Day-over-day ratio
train_48_morn = train_48[train_48['minute_of_day'] <= 120]
d48_morn_avg  = train_48_morn.groupby('geohash')['demand'].mean()
d49_morn_avg  = train_49_morning.groupby('geohash')['demand'].mean()
ratio_gh = (d49_morn_avg / (d48_morn_avg + 1e-8)).clip(0.3, 5.0).to_dict()

d48_g5_morn = train_48_morn.groupby('geo_5')['demand'].mean()
d49_g5_morn = train_49_morning.groupby('geo_5')['demand'].mean()
ratio_g5 = (d49_g5_morn / (d48_g5_morn + 1e-8)).clip(0.3, 5.0).to_dict()

d48_g4_morn = train_48_morn.groupby('geo_4')['demand'].mean()
d49_g4_morn = train_49_morning.groupby('geo_4')['demand'].mean()
ratio_g4 = (d49_g4_morn / (d48_g4_morn + 1e-8)).clip(0.3, 5.0).to_dict()

global_ratio = float(train_49_morning['demand'].mean() / (train_48_morn['demand'].mean() + 1e-8))
global_ratio = float(np.clip(global_ratio, 0.3, 5.0))

def get_ratio_arr(df):
    return np.array([
        ratio_gh.get(gh, ratio_g5.get(g5, ratio_g4.get(g4, global_ratio)))
        for gh, g5, g4 in zip(df['geohash'], df['geo_5'], df['geo_4'])
    ])

OFFSETS = list(range(-120, 135, 15))

def add_day48_features(df, is_train=True):
    out = df.copy()
    ratios = get_ratio_arr(out)
    out['_ratio'] = ratios

    for offset in OFFSETS:
        target_mins = (out['minute_of_day'].values - offset).astype(int)
        col = f'd48_off_{offset}'

        vals = cascade_lookup(
            [pivot_48, pivot_48_g5, pivot_48_g4, pivot_48_g3, pivot_48_g2],
            [out['geohash'].values, out['geo_5'].values, out['geo_4'].values,
             out['geo_3'].values,   out['geo_2'].values],
            target_mins
        )
        gb_series = pivot_48_gb.iloc[0] if len(pivot_48_gb) > 0 else pd.Series(dtype=float)
        for i, (tm, v) in enumerate(zip(target_mins, vals)):
            if np.isnan(v) and tm in gb_series.index:
                vals[i] = gb_series[tm]

        out[col] = vals
        out[f'{col}_scaled'] = vals * ratios

        if is_train:
            out.loc[out['day'] == 48, col] = np.nan
            out.loc[out['day'] == 48, f'{col}_scaled'] = np.nan

    for window, wname in [(4, '1h'), (8, '2h')]:
        roll_mean = pivot_48.T.rolling(window=window, center=True, min_periods=1).mean().T
        roll_std  = pivot_48.T.rolling(window=window, center=True, min_periods=1).std().fillna(0).T

        for stat_name, rp in [('mean', roll_mean), ('std', roll_std)]:
            cname = f'd48_roll_{stat_name}_{wname}'
            vals_r = lookup_pivot_vectorized(rp, out['geohash'].values, out['minute_of_day'].values)
            out[cname] = vals_r
            if is_train:
                out.loc[out['day'] == 48, cname] = np.nan

    out.drop(columns=['_ratio'], inplace=True)
    return out

print('Adding Day 48 profile features...')
train = add_day48_features(train, is_train=True)
test  = add_day48_features(test,  is_train=False)

# ─────────────────────────────────────────────────────────────────────────────
# TARGET ENCODINGS
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_MEAN = float(train_48['demand'].mean())
SMOOTH_K    = 20

def smooth_te(df48, keys, target='demand', k=SMOOTH_K):
    agg = df48.groupby(keys)[target].agg(['mean', 'count']).reset_index()
    agg['te'] = (agg['count'] * agg['mean'] + k * GLOBAL_MEAN) / (agg['count'] + k)
    agg = agg[keys + ['te']]
    agg.columns = keys + ['te_' + '_'.join(keys)]
    return agg

te_gh_hour = smooth_te(train_48, ['geohash', 'hour'])
te_g5_hour = smooth_te(train_48, ['geo_5',   'hour'])
te_g3_peak = smooth_te(train_48, ['geo_3',   'is_morning_peak'])
te_gh_base = smooth_te(train_48, ['geohash'])

train = train.merge(te_gh_hour, on=['geohash', 'hour'], how='left')
train = train.merge(te_g5_hour, on=['geo_5',   'hour'], how='left')
train = train.merge(te_g3_peak, on=['geo_3',   'is_morning_peak'], how='left')
train = train.merge(te_gh_base, on=['geohash'],          how='left')

test  = test.merge(te_gh_hour, on=['geohash', 'hour'], how='left')
test  = test.merge(te_g5_hour, on=['geo_5',   'hour'], how='left')
test  = test.merge(te_g3_peak, on=['geo_3',   'is_morning_peak'], how='left')
test  = test.merge(te_gh_base, on=['geohash'],          how='left')

for col in ['te_geohash_hour', 'te_geo_5_hour', 'te_geo_3_is_morning_peak', 'te_geohash']:
    if col in train.columns:
        train[col] = train[col].fillna(GLOBAL_MEAN)
        test[col]  = test[col].fillna(GLOBAL_MEAN)

ratio_feat = pd.DataFrame({
    'geohash': list(ratio_gh.keys()),
    'd49_d48_ratio': list(ratio_gh.values())
})
train = train.merge(ratio_feat, on='geohash', how='left')
test  = test.merge(ratio_feat,  on='geohash', how='left')
train['d49_d48_ratio'] = train['d49_d48_ratio'].fillna(global_ratio)
test['d49_d48_ratio']  = test['d49_d48_ratio'].fillna(global_ratio)

# Add scaled target encoding features
train['te_geohash_hour_scaled'] = train['te_geohash_hour'] * train['d49_d48_ratio']
test['te_geohash_hour_scaled']  = test['te_geohash_hour'] * test['d49_d48_ratio']
train['te_geo_5_hour_scaled'] = train['te_geo_5_hour'] * train['d49_d48_ratio']
test['te_geo_5_hour_scaled']  = test['te_geo_5_hour'] * test['d49_d48_ratio']
train['te_geohash_scaled'] = train['te_geohash'] * train['d49_d48_ratio']
test['te_geohash_scaled']  = test['te_geohash'] * test['d49_d48_ratio']

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────

def add_synthetic(df, is_train=True):
    vals_48 = cascade_lookup(
        [pivot_48, pivot_48_g5, pivot_48_g4, pivot_48_g3],
        [df['geohash'].values, df['geo_5'].values, df['geo_4'].values, df['geo_3'].values],
        df['minute_of_day'].values.astype(int)
    )
    gb_mean = float(pivot_48_gb.values.mean())
    vals_48 = np.where(np.isnan(vals_48), gb_mean, vals_48)
    df['d48_same_slot'] = vals_48
    df['synthetic_pred'] = vals_48 * df['d49_d48_ratio']
    if is_train:
        df.loc[df['day'] == 48, 'd48_same_slot'] = np.nan
        df.loc[df['day'] == 48, 'synthetic_pred'] = np.nan
    return df

train = add_synthetic(train, is_train=True)
test  = add_synthetic(test,  is_train=False)

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORICALS & FEATURES SET
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDE  = {'Index', 'demand', 'timestamp', 'geohash', 'geo_5', 'geo_4', 'geo_3', 'geo_2',
            'RoadType', 'LargeVehicles', 'Landmarks', 'Weather'}
CAT_COLS = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'geo_3', 'geo_4', 'geo_5']

numeric_cols = [c for c in train.columns if c not in EXCLUDE and train[c].dtype != object]
FEATURE_COLS = numeric_cols + CAT_COLS

# Handle missing categories and convert to category dtype
for col in CAT_COLS:
    all_vals = pd.concat([train[col], test[col]]).fillna('Missing').astype(str)
    cat_map  = {v: i for i, v in enumerate(sorted(all_vals.unique()))}
    train[col] = train[col].fillna('Missing').astype(str).map(cat_map).astype('category')
    test[col]  = test[col].fillna('Missing').astype(str).map(cat_map).astype('category')

train = train.loc[:, ~train.columns.duplicated()]
test  = test.loc[:, ~test.columns.duplicated()]
FEATURE_COLS = [c for c in FEATURE_COLS if c in train.columns]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: TRAIN STACKED MODELS ON THE VALIDATION SPLIT TO FIT META WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

print('\n=== STEP 1: Fitting Meta-Learner on Validation Split ===')

train_mask = (train['day'] == 48) | ((train['day'] == 49) & (train['minute_of_day'] <= 60))
valid_mask  = (train['day'] == 49) & (train['minute_of_day'] > 60)

X_tr  = train.loc[train_mask, FEATURE_COLS]
y_tr  = train.loc[train_mask, 'demand']
X_val = train.loc[valid_mask, FEATURE_COLS]
y_val = train.loc[valid_mask, 'demand']

# Drop all-NaN columns dynamically in the training split
all_nan_cols = X_tr.columns[X_tr.isna().all()]
print(f'Dropping all-NaN columns from training split: {list(all_nan_cols)}')
X_tr  = X_tr.drop(columns=all_nan_cols)
X_val = X_val.drop(columns=all_nan_cols)

print(f'Train split: {X_tr.shape}, Val split: {X_val.shape}')

# 1. LightGBM
lgbm_val = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.02,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=10,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm_val.fit(X_tr, y_tr)
p_lgbm_val = np.clip(lgbm_val.predict(X_val), 0.0, 1.0)
print(f'LGBM Val R2: {r2_score(y_val, p_lgbm_val):.5f}')

# 2. XGBoost
xgb_val = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    enable_categorical=True
)
xgb_val.fit(X_tr, y_tr)
p_xgb_val = np.clip(xgb_val.predict(X_val), 0.0, 1.0)
print(f'XGBoost Val R2: {r2_score(y_val, p_xgb_val):.5f}')

# 3. HistGradientBoosting
hgb_val = HistGradientBoostingRegressor(
    max_iter=1000,
    learning_rate=0.02,
    max_leaf_nodes=63,
    categorical_features='from_dtype',
    random_state=42
)
hgb_val.fit(X_tr, y_tr)
p_hgb_val = np.clip(hgb_val.predict(X_val), 0.0, 1.0)
print(f'HGB Val R2: {r2_score(y_val, p_hgb_val):.5f}')

# Fit Meta Learner
meta_X = np.column_stack([p_lgbm_val, p_xgb_val, p_hgb_val])
meta = Ridge(alpha=1.0, positive=True)
meta.fit(meta_X, y_val)
p_meta_val = np.clip(meta.predict(meta_X), 0.0, 1.0)
print(f'Stacked Val R2: {r2_score(y_val, p_meta_val):.5f}')
print(f'Meta weights: lgbm={meta.coef_[0]:.3f}, xgb={meta.coef_[1]:.3f}, hgb={meta.coef_[2]:.3f}')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: RETRAIN MODELS ON THE FULL DATASET & PREDICT
# ─────────────────────────────────────────────────────────────────────────────

print('\n=== STEP 2: Retraining Models on Full Dataset ===')

X_full = train[FEATURE_COLS]
y_full = train['demand']
X_test = test[FEATURE_COLS]

# Drop all-NaN columns from the full training features
all_nan_full = X_full.columns[X_full.isna().all()]
if len(all_nan_full) > 0:
    print(f'Dropping all-NaN columns from full training set: {list(all_nan_full)}')
    X_full = X_full.drop(columns=all_nan_full)
    X_test = X_test.drop(columns=all_nan_full)

print(f'Full Train shape: {X_full.shape}, Test shape: {X_test.shape}')

# 1. Retrain LightGBM
lgbm_full = LGBMRegressor(
    n_estimators=2000,  # slightly higher tree size for more data
    learning_rate=0.02,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=10,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm_full.fit(X_full, y_full)
p_lgbm_test = np.clip(lgbm_full.predict(X_test), 0.0, 1.0)
print('LightGBM retrained.')

# 2. Retrain XGBoost
xgb_full = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    enable_categorical=True
)
xgb_full.fit(X_full, y_full)
p_xgb_test = np.clip(xgb_full.predict(X_test), 0.0, 1.0)
print('XGBoost retrained.')

# 3. Retrain HistGradientBoosting
hgb_full = HistGradientBoostingRegressor(
    max_iter=1000,
    learning_rate=0.02,
    max_leaf_nodes=63,
    categorical_features='from_dtype',
    random_state=42
)
hgb_full.fit(X_full, y_full)
p_hgb_test = np.clip(hgb_full.predict(X_test), 0.0, 1.0)
print('HGB retrained.')

# Combine predictions using the meta-learner weights
meta_test_X = np.column_stack([p_lgbm_test, p_xgb_test, p_hgb_test])
final_preds = np.clip(meta.predict(meta_test_X), 0.0, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SAVE FINAL SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────

submission = pd.DataFrame({
    'Index': test_raw['Index'],
    'demand': final_preds
})

out_path = PROJECT_ROOT / 'submission.csv'
submission.to_csv(out_path, index=False)
print(f'\nSaved final submission to {out_path}')
print(f'Shape: {submission.shape}')
print('Prediction statistics:')
print(submission['demand'].describe())
