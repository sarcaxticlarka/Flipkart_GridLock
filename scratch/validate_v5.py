import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from pathlib import Path

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path('/Users/underxcore/Desktop/Flipkart_GridLock')
DATA_DIR     = PROJECT_ROOT / 'dataset'

train_raw = pd.read_csv(DATA_DIR / 'train.csv')
test_raw  = pd.read_csv(DATA_DIR / 'test.csv')

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

train_49 = train[train['day'] == 49]
train_49_geo = train_49[['geo_5','geo_4','geo_3','geo_2','minute_of_day','demand']]

pivot_49      = train_49.pivot_table(index='geohash', columns='minute_of_day', values='demand', aggfunc='mean')
pivot_49_g5   = train_49_geo.pivot_table(index='geo_5',  columns='minute_of_day', values='demand', aggfunc='mean')
pivot_49_g4   = train_49_geo.pivot_table(index='geo_4',  columns='minute_of_day', values='demand', aggfunc='mean')
pivot_49_g3   = train_49_geo.pivot_table(index='geo_3',  columns='minute_of_day', values='demand', aggfunc='mean')
pivot_49_g2   = train_49_geo.pivot_table(index='geo_2',  columns='minute_of_day', values='demand', aggfunc='mean')
pivot_49_gb   = train_49_geo.pivot_table(columns='minute_of_day', values='demand', aggfunc='mean')

avail_minutes_49 = sorted(train_49['minute_of_day'].unique())

def add_day49_lag_features(df, is_train=True):
    out = df.copy()
    for lag_min in avail_minutes_49:
        col_name = f'd49_lag_{lag_min}'
        lag_col  = np.full(len(out), lag_min)

        vals = cascade_lookup(
            [pivot_49, pivot_49_g5, pivot_49_g4, pivot_49_g3, pivot_49_g2],
            [out['geohash'].values, out['geo_5'].values, out['geo_4'].values,
             out['geo_3'].values, out['geo_2'].values],
            lag_col.astype(int)
        )
        gb_val = float(pivot_49_gb[lag_min].iloc[0]) if lag_min in pivot_49_gb.columns else np.nan
        vals = np.where(np.isnan(vals), gb_val, vals)
        out[col_name] = vals

        if is_train:
            out.loc[(out['day'] == 49) & (out['minute_of_day'] <= lag_min), col_name] = np.nan
    return out

print('Adding Day 49 lag features...')
train = add_day49_lag_features(train, is_train=True)
test  = add_day49_lag_features(test,  is_train=False)

train_49_morning = train[(train['day'] == 49) & (train['minute_of_day'] <= 120)]
gh_agg = train_49_morning.groupby('geohash')['demand'].agg(
    d49_morning_mean='mean',
    d49_morning_std='std',
    d49_morning_max='max',
    d49_morning_min='min',
).reset_index()
gh_agg['d49_morning_std'] = gh_agg['d49_morning_std'].fillna(0)

vel_rows = []
for gh, grp in train_49_morning.sort_values('minute_of_day').groupby('geohash'):
    if len(grp) >= 2:
        slope = float(np.polyfit(grp['minute_of_day'].values.astype(float), grp['demand'].values, 1)[0])
    else:
        slope = 0.0
    vel_rows.append({'geohash': gh, 'd49_velocity': slope})
velocity_df = pd.DataFrame(vel_rows)
gh_agg = gh_agg.merge(velocity_df, on='geohash', how='left')

g5_tmp = train_49_morning.copy()
g5_tmp['geo_5'] = g5_tmp['geohash'].str[:5]
g5_agg = g5_tmp.groupby('geo_5')['demand'].agg(
    d49_morning_mean_g5='mean',
    d49_velocity_g5='mean'
).reset_index()

train = train.merge(gh_agg, on='geohash', how='left')
train = train.merge(g5_agg, on='geo_5',   how='left')
test  = test.merge(gh_agg,  on='geohash', how='left')
test  = test.merge(g5_agg,  on='geo_5',   how='left')

for df in [train, test]:
    df['d49_morning_mean'] = df['d49_morning_mean'].fillna(df['d49_morning_mean_g5'])
    df['d49_velocity']     = df['d49_velocity'].fillna(df['d49_velocity_g5']).fillna(0)
    df['d49_morning_std']  = df['d49_morning_std'].fillna(0)

train_48 = train[train['day'] == 48]
train_48_geo = train_48[['geo_5','geo_4','geo_3','geo_2','minute_of_day','demand']]

pivot_48    = train_48.pivot_table(index='geohash', columns='minute_of_day', values='demand', aggfunc='mean')
pivot_48_g5 = train_48_geo.pivot_table(index='geo_5', columns='minute_of_day', values='demand', aggfunc='mean')
pivot_48_g4 = train_48_geo.pivot_table(index='geo_4', columns='minute_of_day', values='demand', aggfunc='mean')
pivot_48_g3 = train_48_geo.pivot_table(index='geo_3', columns='minute_of_day', values='demand', aggfunc='mean')
pivot_48_g2 = train_48_geo.pivot_table(index='geo_2', columns='minute_of_day', values='demand', aggfunc='mean')
pivot_48_gb = train_48_geo.pivot_table(columns='minute_of_day', values='demand', aggfunc='mean')

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

EXCLUDE  = {'Index', 'demand', 'timestamp', 'geohash', 'geo_5', 'geo_4', 'geo_3', 'geo_2',
            'RoadType', 'LargeVehicles', 'Landmarks', 'Weather',
            'd49_morning_mean_g5', 'd49_velocity_g5'}
CAT_COLS = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'geo_3', 'geo_4', 'geo_5']

numeric_cols = [c for c in train.columns if c not in EXCLUDE and train[c].dtype != object]
FEATURE_COLS = numeric_cols + CAT_COLS

# FIX HERE: Fillna with 'Missing' first to ensure everything is a string and sorted() doesn't complain about floats (like nan)
for col in CAT_COLS:
    all_vals = pd.concat([train[col], test[col]]).fillna('Missing').astype(str)
    cat_map  = {v: i for i, v in enumerate(sorted(all_vals.unique()))}
    train[col] = train[col].fillna('Missing').astype(str).map(cat_map).fillna(-1).astype(int)
    test[col]  = test[col].fillna('Missing').astype(str).map(cat_map).fillna(-1).astype(int)

train = train.loc[:, ~train.columns.duplicated()]
test  = test.loc[:, ~test.columns.duplicated()]
FEATURE_COLS = [c for c in FEATURE_COLS if c in train.columns]

# Train on Day 48 (all) + Day 49 minutes 0-60; validate on Day 49 minutes 75-120
train_mask = (train['day'] == 48) | ((train['day'] == 49) & (train['minute_of_day'] <= 60))
valid_mask  = (train['day'] == 49) & (train['minute_of_day'] > 60)

X_tr  = train.loc[train_mask, FEATURE_COLS]
y_tr  = train.loc[train_mask, 'demand']
X_val = train.loc[valid_mask, FEATURE_COLS]
y_val = train.loc[valid_mask, 'demand']

imp = SimpleImputer(strategy='median')
X_tr_imp  = pd.DataFrame(imp.fit_transform(X_tr),  columns=FEATURE_COLS)
X_val_imp = pd.DataFrame(imp.transform(X_val),     columns=FEATURE_COLS)

print('X_tr_imp shape:', X_tr_imp.shape)
print('X_val_imp shape:', X_val_imp.shape)

lgbm = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_tr_imp, y_tr, eval_set=[(X_val_imp, y_val)])
p_lgbm = np.clip(lgbm.predict(X_val_imp), 0, 1)
r2_lgbm = r2_score(y_val, p_lgbm)
print(f'LightGBM Val R2: {r2_lgbm:.5f} -> score: {100*r2_lgbm:.2f}')

xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=100,
    eval_metric='rmse'
)
xgb.fit(X_tr_imp, y_tr, eval_set=[(X_val_imp, y_val)], verbose=False)
p_xgb = np.clip(xgb.predict(X_val_imp), 0, 1)
r2_xgb = r2_score(y_val, p_xgb)
print(f'XGBoost Val R2: {r2_xgb:.5f} -> score: {100*r2_xgb:.2f}')

hgb = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.03,
    max_leaf_nodes=31,
    random_state=42
)
hgb.fit(X_tr_imp, y_tr)
p_hgb = np.clip(hgb.predict(X_val_imp), 0, 1)
r2_hgb = r2_score(y_val, p_hgb)
print(f'HGB Val R2: {r2_hgb:.5f} -> score: {100*r2_hgb:.2f}')

p_ens = np.clip(0.50 * p_lgbm + 0.35 * p_xgb + 0.15 * p_hgb, 0, 1)
r2_ens = r2_score(y_val, p_ens)
print(f'Weighted Ensemble Val R2: {r2_ens:.5f} -> score: {100*r2_ens:.2f}')

meta_X = np.column_stack([p_lgbm, p_xgb, p_hgb])
meta = Ridge(alpha=1.0, positive=True)
meta.fit(meta_X, y_val)
p_meta = np.clip(meta.predict(meta_X), 0, 1)
r2_meta = r2_score(y_val, p_meta)
print(f'Stacked Val R2: {r2_meta:.5f} -> score: {100*r2_meta:.2f}')
