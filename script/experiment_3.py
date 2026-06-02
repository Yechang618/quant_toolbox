import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import scipy.stats as stats

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import xgboost as xgb
# import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import seaborn as sns

results = {}
models = ['OLS Regression', 'Linear Regression',  
        #   'LightGBM Regressor', #'XGBoost Regressor', 
          # 'Random Forest Regressor',  #'CNN Regressor', 
          ]

TOLERENCE = 1e-12
# mode = 2
delay_exec = ''
normalize_X = 0
# mode, operation = 2, 'close2'
# mode, operation = 2, 'open2'
mode, operation = 0, 'close2'
# mode, operation = 0, 'open2'
delay_precentile = 95
beta = 1
symbol = 'all'
# symbol = 'ZENUSDT'

######### label
label_name = 'gain_vs_threshold' 

# Define the folder path
folder_path = f'data/factors_output/mode{mode}/'


# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Read and combine all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)
print(f"Combined {len(csv_files)} files")
print(f"Total rows: {len(combined_df)}")
print(combined_df.info(verbose=True, show_counts=True))
print(f"List of feature columns: {combined_df.columns.tolist()}")

selected_cols = ['gain_vs_threshold', 'basis_slippage', 'symbol', 'trade_mode', 
                 'operation', 'exec_ts_utc', 'execute_delay_ms', 'threshold', 
                 'basis_expected', 'basis_executed', 
                 'baa_-8_sum_ws', 'baa_-8_sum_w', 'baa_-7_sum_ws', 'baa_-7_sum_w', 
                 'baa_-6_sum_ws', 'baa_-6_sum_w', 'baa_-5_sum_ws', 'baa_-5_sum_w', 
                 'baa_-4_sum_ws', 'baa_-4_sum_w', 'baa_-3_sum_ws', 'baa_-3_sum_w', 
                 'baa_-2_sum_ws', 'baa_-2_sum_w', 'baa_-1_sum_ws', 'baa_-1_sum_w', 
                 'baa_0_sum_ws', 'baa_0_sum_w', 
                 'baa_1_sum_ws', 'baa_1_sum_w', 'baa_2_sum_ws', 'baa_2_sum_w', 
                 'baa_3_sum_ws', 'baa_3_sum_w', 'baa_4_sum_ws', 'baa_4_sum_w', 
                 'baa_5_sum_ws', 'baa_5_sum_w', 'baa_6_sum_ws', 'baa_6_sum_w', 
                 'baa_7_sum_ws', 'baa_7_sum_w', 'baa_8_sum_ws', 'baa_8_sum_w', 
                 'bbb_-8_sum_ws', 'bbb_-8_sum_w', 'bbb_-7_sum_ws', 'bbb_-7_sum_w', 
                 'bbb_-6_sum_ws', 'bbb_-6_sum_w', 'bbb_-5_sum_ws', 'bbb_-5_sum_w', 
                 'bbb_-4_sum_ws', 'bbb_-4_sum_w', 'bbb_-3_sum_ws', 'bbb_-3_sum_w', 
                 'bbb_-2_sum_ws', 'bbb_-2_sum_w', 'bbb_-1_sum_ws', 'bbb_-1_sum_w', 
                 'bbb_0_sum_ws', 'bbb_0_sum_w', 'bbb_1_sum_ws', 'bbb_1_sum_w', 
                 'bbb_2_sum_ws', 'bbb_2_sum_w', 'bbb_3_sum_ws', 'bbb_3_sum_w', 
                 'bbb_4_sum_ws', 'bbb_4_sum_w', 'bbb_5_sum_ws', 'bbb_5_sum_w', 
                 'bbb_6_sum_ws', 'bbb_6_sum_w', 'bbb_7_sum_ws', 'bbb_7_sum_w', 
                 'bbb_8_sum_ws', 'bbb_8_sum_w', 'bab_-8_sum_ws', 'bab_-8_sum_w', 
                 'bab_-7_sum_ws', 'bab_-7_sum_w', 'bab_-6_sum_ws', 'bab_-6_sum_w', 
                 'bab_-5_sum_ws', 'bab_-5_sum_w', 'bab_-4_sum_ws', 'bab_-4_sum_w', 
                 'bab_-3_sum_ws', 'bab_-3_sum_w', 'bab_-2_sum_ws', 'bab_-2_sum_w', 
                 'bab_-1_sum_ws', 'bab_-1_sum_w', 'bab_0_sum_ws', 'bab_0_sum_w', 
                 'bab_1_sum_ws', 'bab_1_sum_w', 'bab_2_sum_ws', 'bab_2_sum_w', 
                 'bab_3_sum_ws', 'bab_3_sum_w', 'bab_4_sum_ws', 'bab_4_sum_w', 
                 'bab_5_sum_ws', 'bab_5_sum_w', 'bab_6_sum_ws', 'bab_6_sum_w', 
                 'bab_7_sum_ws', 'bab_7_sum_w', 'bab_8_sum_ws', 'bab_8_sum_w', 
                 'bba_-8_sum_ws', 'bba_-8_sum_w', 'bba_-7_sum_ws', 'bba_-7_sum_w', 
                 'bba_-6_sum_ws', 'bba_-6_sum_w', 'bba_-5_sum_ws', 'bba_-5_sum_w', 
                 'bba_-4_sum_ws', 'bba_-4_sum_w', 'bba_-3_sum_ws', 'bba_-3_sum_w', 
                 'bba_-2_sum_ws', 'bba_-2_sum_w', 'bba_-1_sum_ws', 'bba_-1_sum_w', 
                 'bba_0_sum_ws', 'bba_0_sum_w', 'bba_1_sum_ws', 'bba_1_sum_w', 
                 'bba_2_sum_ws', 'bba_2_sum_w', 'bba_3_sum_ws', 'bba_3_sum_w', 
                 'bba_4_sum_ws', 'bba_4_sum_w', 'bba_5_sum_ws', 'bba_5_sum_w', 
                 'bba_6_sum_ws', 'bba_6_sum_w', 'bba_7_sum_ws', 'bba_7_sum_w', 
                 'bba_8_sum_ws', 'bba_8_sum_w']
weighted_feature_cols = [col for col in selected_cols if 'basis_mid_weighted_mean' in col]
# Obtain weights from the column names
weights = []
for col in weighted_feature_cols:
    weight_str = col.replace('basis_mid_weighted_mean_', '')
    weights.append(weight_str)

print(f"Extracted weights: {weights}")


# # Sort by exec_ts_utc
# # combined_df['exec_ts_utc'] = pd.to_datetime(combined_df['exec_ts_utc'])
# combined_df['exec_ts_utc'] = pd.to_datetime(
#     combined_df['exec_ts_utc'],
#     format='ISO8601',
#     utc=True
# )
# 🔧 修复：毫秒时间戳需指定 unit='ms'，移除冗余的重复转换
combined_df['exec_ts_utc'] = pd.to_datetime(combined_df['exec_ts_utc'], unit='ms', utc=True, errors='coerce')
combined_df = combined_df.sort_values('exec_ts_utc')
# combined_df = combined_df.sort_values('exec_ts_utc')

# print(combined_df['execute_delay_ms'].describe())

# operation_select = 'open2'
# operation_select = 'close2'
# delay_quantile = delay_precentile # Filter out rows with execute_delay_ms above the delay_precentile (median) to focus on more typical cases. Adjust as needed (e.g., 80 for 80th percentile).
# Filter out outliers based on the 95th percentile of execute_delay_ms
upper_limit_delay = combined_df['execute_delay_ms'].quantile(delay_precentile/100)
print(f"{delay_precentile}th percentile of execute_delay_ms: {upper_limit_delay} ms")
filtered_df = combined_df[combined_df['execute_delay_ms'] <= upper_limit_delay]

# Filter out outliers based on the 10th and 90th percentiles of gain_vs_threshold
lower_limit_gain = filtered_df['gain_vs_threshold'].quantile(0.05)
upper_limit_gain = filtered_df['gain_vs_threshold'].quantile(0.95)
print(f"5th percentile of gain_vs_threshold: {lower_limit_gain}")
print(f"95th percentile of gain_vs_threshold: {upper_limit_gain}")
filtered_df = filtered_df[(filtered_df['gain_vs_threshold'] >= lower_limit_gain) & (filtered_df['gain_vs_threshold'] <= upper_limit_gain)]
filtered_df = filtered_df[(filtered_df['operation'] == operation)]
if symbol != 'all':
    filtered_df = filtered_df[filtered_df['symbol'] == symbol]
print(f"Operation: {operation}, Remaining rows after filtering: {len(filtered_df)}")

feature_cols = [
                # 'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean', 'spot_depth5_imbalance_mean', 'swap_depth5_imbalance_mean', 
                # 'swap_buy_trade_ratio', 'spot_buy_trade_ratio',
                ]


selected_feature_cols = []
# # ✅ 替换为：批量构建字典后一次性 concat（零碎片化）

# ✅ 替换原 dropna 逻辑：仅剔除核心标签/元数据为空的行，保留特征中的 NaN
critical_cols = ['gain_vs_threshold', 'threshold', 'basis_expected', 'basis_executed', 'exec_ts_utc']
# 安全过滤：只保留 selected_cols 中实际存在于 df 的列
existing_cols = [c for c in selected_cols if c in filtered_df.columns]
df = filtered_df[existing_cols].dropna(subset=critical_cols)

if df.empty:
    print("❌ 致命错误: 过滤后数据集为空！请检查 selected_cols 是否含拼写错误（如空格）或筛选条件过严。")
    exit(1)
print(f"Final dataset shape after filtering: {df.shape}")

# ✅ 替换原逐列赋值逻辑：使用字典批量构建，彻底消除 PerformanceWarning
new_cols_dict = {}

if 'basis_ask_mean' in df.columns and 'basis_bid_mean' in df.columns:
    new_cols_dict['basis_mid_mean'] = (df['basis_ask_mean'] + df['basis_bid_mean']) / 2
    new_cols_dict['basis_mid_to_thres'] = new_cols_dict['basis_mid_mean'] - df['threshold']
    feature_cols.append('basis_mid_to_thres')

# 
basis_cols = {'baa':['bba', 'bab'], 'bbb':['bab', 'bba']}
alphas = [.5, 1, 2]
N_weights = 8
# b1 = 'baa', 'bbb'
# b1 + alpha * b2
for b1 in basis_cols:
    b2_list = basis_cols[b1]
    for w in range(-N_weights, N_weights+1):
        new_col_name = f'BWT_{b1}_{w}'
        new_cols_dict[new_col_name] = df[f'{b1}_{w}_sum_ws'] / (df[f'{b1}_{w}_sum_w'] + 1e-10) - df['threshold']
        feature_cols.append(new_col_name)
    for b2 in b2_list:
        for alpha in alphas:
            for w in range(-N_weights, N_weights+1):
                # Method 1
                new_col_name = f'BWT_{b1}_{b2}_{w}_{alpha}'
                new_cols_dict[new_col_name] = (df[f'{b1}_{w}_sum_ws'] +  alpha * df[f'{b2}_{w}_sum_ws']) / (df[f'{b1}_{w}_sum_w'] + alpha * df[f'{b2}_{w}_sum_w']) - df['threshold']
                feature_cols.append(new_col_name)
                # Method 2 (alternative weighting scheme)
                new_col_name_alt = f'BWT2_{b1}_{b2}_{w}_{alpha}'
                new_cols_dict[new_col_name_alt] = (df[f'{b1}_{w}_sum_ws']/(df[f'{b1}_{w}_sum_w']+1e-10)  +  alpha * df[f'{b2}_{w}_sum_ws']/(df[f'{b2}_{w}_sum_w']+1e-10)) / (1 + alpha) - df['threshold']
                feature_cols.append(new_col_name_alt)

# b1 = 'bba'
# w = 0
# new_col_name = f'BWT_{b1}_{w}'
# new_cols_dict[new_col_name] = df[f'{b1}_{w}_sum_ws'] / (df[f'{b1}_{w}_sum_w'] + 1e-10) - df['threshold']
# feature_cols.append(new_col_name)
# b1 = 'bab'
# w = 0
# new_col_name = f'BWT_{b1}_{w}'
# new_cols_dict[new_col_name] = df[f'{b1}_{w}_sum_ws'] / (df[f'{b1}_{w}_sum_w'] + 1e-10) - df['threshold']
# feature_cols.append(new_col_name)
for b1 in ['bba', 'bab']:
    for w in [-5, 0, 5]:
        new_col_name = f'BWT_{b1}_{w}'
        new_cols_dict[new_col_name] = df[f'{b1}_{w}_sum_ws'] / (df[f'{b1}_{w}_sum_w'] + 1e-10) - df['threshold']
        feature_cols.append(new_col_name)

# 一次性追加所有新列到 DataFrame
if new_cols_dict:
    df = pd.concat([df, pd.DataFrame(new_cols_dict, index=df.index)], axis=1)

df = df.copy()

# 2. 创建日期列（使用 floor 保持 datetime64 类型）
df['date'] = df['exec_ts_utc'].dt.floor('D')

# Compute daily Absolute error and Squared Error for the label
from scipy.optimize import minimize
# Define the objective function |ax - y|

def calc_ae(df, col, label_name):
    valid_idx = df[[col, label_name]].dropna().index
    if len(valid_idx) < 3:
        return np.nan
    x = df.loc[valid_idx, col]
    y = df.loc[valid_idx, label_name]
    def objective(a):
        return np.sum(np.abs(a * x - y))
    # x = df['BWT_b1_b2_w_alpha'] 
    # y = df['gain_vs_threshold']
    # y_hat = a * x
    # error = y - y_hat
    # Return |error| = |y_hat - ax|
    # min_a |a*x - y| => a
    res = minimize(objective, x0=1.0, method='Nelder-Mead')
    if not res.success:
        print(f"Failed to find optimal a for {col}: {res.x[0]:.4f}, AE: {res.fun:.4f}")
    return res.fun / len(valid_idx) 

def calc_diff(df, col, label_name, op = 'open2'):
    valid_idx = df[[col, label_name]].dropna().index
    if len(valid_idx) < 3:
        return np.nan
    x = df.loc[valid_idx, col]
    y = df.loc[valid_idx, label_name]
    if op == 'close2':
        y = -y
    res = sum(abs(y - x))
    # if not res.success:
    #     print(f"Failed to find optimal a for {col}: {res.x[0]:.4f}, AE: {res.fun:.4f}")
    return res / len(valid_idx) 

# Compute IC/IR/AE(Absolute Error) for feature_cols
for col in feature_cols:
    if col not in df.columns:
        print(f"⚠️ 警告: 计算 IC/IR 时缺失列 {col}，将跳过该列的计算。")
        continue
    def calc_daily_ic(group):
        valid_idx = group[[col, label_name]].dropna().index
        if len(valid_idx) < 3:
            return np.nan
        x = group.loc[valid_idx, col]
        y = group.loc[valid_idx, label_name]
        return stats.pearsonr(x, y)[0]
    daily_ic = df.groupby('date').apply(calc_daily_ic)
    daily_ic = pd.to_numeric(daily_ic, errors='coerce')
    # daily_ae = df.groupby('date').apply(lambda g: mean_squared_error(g[label_name], g[col], squared=False))
    # ✅ 替换为（安全计算 RMSE，自动处理 NaN）
    ae = calc_ae(df, col, label_name)
    ae = pd.to_numeric(ae, errors='coerce')
    ab_dif = calc_diff(df, col, label_name, op = operation)
    ab_dif = pd.to_numeric(ab_dif, errors='coerce')
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    ir = ic_mean / ic_std if pd.notna(ic_std) and ic_std > 0 else np.nan
    print(f"✅ IC of {col}: {ic_mean:.8f} (±{ic_std:.8f}) | IR: {ir:.8f} | AE: {ae:.8f} | AD: {ab_dif:.8f} ")

# 3. 定义要计算 IC/IR 的因子列表
ic_ir_list = feature_cols
ic_ir_list_single = [col for col in ic_ir_list if 'BWT' in col]
res_ic_ir_single = np.zeros((len(ic_ir_list_single), 3))

print(f"ICIR list single: {ic_ir_list_single}")
print(f"Total features to calculate IC/IR: {len(ic_ir_list_single)}")
# 如果还需要计算原始 mid 因子的 IC，可取消下一行注释

for i, factor_name in enumerate(ic_ir_list_single):
    if factor_name not in df.columns:
        print(f"⚠️ 跳过缺失因子: {factor_name}")
        continue

    def calc_daily_ic(group):
        # 🛡️ 核心修复：基于共同非空索引对齐因子与标签，确保 len(x) == len(y)
        valid_idx = group[[factor_name, label_name]].dropna().index
        if len(valid_idx) < 3:  # 样本过少时 spearmanr 无统计意义
            return np.nan
        x = group.loc[valid_idx, factor_name]
        y = group.loc[valid_idx, label_name]
        return stats.pearsonr(x, y)[0]

    # 计算每日 IC
    daily_ic = df.groupby('date').apply(calc_daily_ic)
    daily_ic = pd.to_numeric(daily_ic, errors='coerce')  # 强制转 float 防 object 类型报错
    # ✅ 替换为（修正拼写 + 兼容新版 sklearn）
    # ae = calc_ae(df, factor_name, label_name)
    # ae = pd.to_numeric(ae, errors='coerce')
    ab_dif = calc_diff(df, factor_name, label_name, op = operation)
    ab_dif = pd.to_numeric(ab_dif, errors='coerce')

    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    ir = ic_mean / ic_std if pd.notna(ic_std) and ic_std > 0 else np.nan

    # print(f"✅ IC of {factor_name}: {ic_mean:.8f} (±{ic_std:.8f}) | IR: {ir:.8f}")
    res_ic_ir_single[i, 0] = ic_mean
    res_ic_ir_single[i, 1] = ir
    # res_ic_ir_single[i, 2] = ae
    res_ic_ir_single[i, 2] = ab_dif
# Plot 10 features with highest IC values
ic_values = res_ic_ir_single[:, 0]
top_10_features = sorted(zip(ic_ir_list_single, ic_values), key=lambda x: x[1], reverse=True)[:10]
top_10_feature_names = [name for name, _ in top_10_features]
top_10_ic_values = [ic for _, ic in top_10_features]
top_all_features = sorted(zip(ic_ir_list_single, ic_values), key=lambda x: x[1], reverse=True)
top_all_feature_names = [name for name, _ in top_all_features]
top_all_ic_values = [ic for _, ic in top_all_features]
print("Top 10 features by IC:")
for name, ic in top_10_features:
    print(f"{name}: IC={ic:.8f}")
bottom_10_features = sorted(zip(ic_ir_list_single, ic_values), key=lambda x: x[1])[:10]
bottom_10_feature_names = [name for name, _ in bottom_10_features]
bottom_10_ic_values = [ic for _, ic in bottom_10_features]
print("Bottom 10 features by IC:")
for name, ic in bottom_10_features:
    print(f"{name}: IC={ic:.8f}")
# Plot 10 features with highest IR values
ir_values = res_ic_ir_single[:, 1]
top_10_ir_features = sorted(zip(ic_ir_list_single, ir_values), key=lambda x: x[1], reverse=True)[:10]
top_10_ir_feature_names = [name for name, _ in top_10_ir_features]
top_10_ir_values = [ir for _, ir in top_10_ir_features]
print("Top 10 features by IR:")
for name, ir in top_10_ir_features:
    print(f"{name}: IR={ir:.8f}")
bottom_10_ir_features = sorted(zip(ic_ir_list_single, ir_values), key=lambda x: x[1])[:10]
bottom_10_ir_feature_names = [name for name, _ in bottom_10_ir_features]
bottom_10_ir_values = [ir for _, ir in bottom_10_ir_features]
print("Bottom 10 features by IR:")
for name, ir in bottom_10_ir_features:
    print(f"{name}: IR={ir:.8f}")
# Plot 10 features with highest AE values
# ae_values = res_ic_ir_single[:, 2]
# top_10_ae_features = sorted(zip(ic_ir_list_single, ae_values), key=lambda x: x[1])[:10]
# top_10_ae_feature_names = [name for name, _ in top_10_ae_features]
# top_10_ae_values = [ae for _, ae in top_10_ae_features]
# print("Top 10 features by AE (lower is better):")
# for name, ae in top_10_ae_features:
#     print(f"{name}: AE={ae:.8f}")
# Plot 10 features with lowest ab_dif values
ab_dif_values = res_ic_ir_single[:, 2]
top_10_ab_dif_features = sorted(zip(ic_ir_list_single, ab_dif_values), key=lambda x: x[1])
# Remove Inf and NaN from top_10_ab_dif_features
top_10_ab_dif_features = [(name, ab_dif) for name, ab_dif in top_10_ab_dif_features if pd.notna(ab_dif) and np.isfinite(ab_dif)]
top_10_ab_dif_features  = top_10_ab_dif_features [:10]
print(f"top 10: {top_10_ab_dif_features[:10]}")
top_10_ab_dif_feature_names = [name for name, _ in top_10_ab_dif_features]
top_10_ab_dif_values = [ab_dif for _, ab_dif in top_10_ab_dif_features]
print("Top 10 features by Absolute Difference (lower is better):")
for name, ab_dif in top_10_ab_dif_features:
    print(f"{name}: AD={ab_dif:.8f}")


# bottom_10_ae_features = sorted(zip(ic_ir_list_single, ae_values), key=lambda x: x[1], reverse=True)[:10]
# bottom_10_ae_feature_names = [name for name, _ in bottom_10_ae_features]
# bottom_10_ae_values = [ae for _, ae in bottom_10_ae_features]
# print("Bottom 10 features by AE (higher is worse):")
# for name, ae in bottom_10_ae_features:
#     print(f"{name}: AE={ae:.8f}")
    


# Plotting using matplotlib, subplots for IC and IR
fig, axes = plt.subplots(2, 2, figsize=(24, 24))
# IC plot
axes[0, 0].bar(top_10_feature_names, top_10_ic_values, color='green', label='Top 10 IC')
axes[0, 0].bar(bottom_10_feature_names, bottom_10_ic_values, color='red', label='Bottom 10 IC')
axes[0, 0].set_xticklabels(top_10_feature_names + bottom_10_feature_names, rotation=90)
axes[0, 0].set_xlabel('Features')
axes[0, 0].set_ylabel('IC')
axes[0, 0].set_title('Top/Bottom 10 Features by IC')
axes[0, 0].grid(True); axes[0, 0].legend()
# IR plot
axes[0, 1].bar(top_10_ir_feature_names, top_10_ir_values, color='blue', label='Top 10 IR')
axes[0, 1].bar(bottom_10_ir_feature_names, bottom_10_ir_values, color='orange', label='Bottom 10 IR')
axes[0, 1].set_xticklabels(top_10_ir_feature_names + bottom_10_ir_feature_names, rotation=90)
axes[0, 1].set_xlabel('Features')
axes[0, 1].set_ylabel('IR')
axes[0, 1].set_title('Top/Bottom 10 Features by IR')
axes[0, 1].grid(True); axes[0, 1].legend()
# AE plot
# axes[1, 0].bar(top_10_ae_feature_names, top_10_ae_values, color='purple', label='Top 10 AE')
# # axes[1, 0].bar(bottom_10_ae_feature_names, bottom_10_ae_values, color='yellow', label='Bottom 10 AE')
# axes[1, 0].set_xticklabels(top_10_ae_feature_names, rotation=90)
# axes[1, 0].set_xlabel('Features')
# axes[1, 0].set_ylabel('AE')
# axes[1, 0].set_title('10 Features w. smallest AE')
# axes[1, 0].grid(True); axes[1, 0].legend()
# AD plot
axes[1,1].bar(top_10_ab_dif_feature_names, top_10_ab_dif_values, color='cyan', label='Top 10 AD')
# axes[1,1].bar(bottom_10_ab_dif_feature_names, bottom_10_ab_dif_values, color='magenta', label='Bottom 10 AD')
axes[1,1].set_xticklabels(top_10_ab_dif_feature_names, rotation=90)
axes[1,1].set_xlabel('Features')
axes[1,1].set_ylabel('Absolute Difference')
axes[1,1].set_title('10 Features w. smallest Absolute Difference')
axes[1,1].grid(True); axes[1,1].legend()
plt.tight_layout()
os.makedirs('output', exist_ok=True)
plt.savefig(f'output/res_ic_ir_diff_{symbol}_{label_name}_delay{delay_precentile}_{operation}_mode{mode}.png', bbox_inches='tight')
plt.close(fig)
print(f"✓ Saved combined IC/IR/AD plot → res_ic_ir_diff_*.png")
# =============================================================================
# 📊 独立绘制并保存三个图表：IC / IR / Absolute Difference
# =============================================================================
os.makedirs('output', exist_ok=True)

# -----------------------------------------------------------------------------
# 1️⃣ IC Plot: Top/Bottom 10 Features by Information Coefficient
# -----------------------------------------------------------------------------
fig_ic, ax_ic = plt.subplots(figsize=(14, 8))
all_ic_names = top_10_feature_names + bottom_10_feature_names
n_top = len(top_10_feature_names)
n_bottom = len(bottom_10_feature_names)

# ✅ 核心修复：分开绘制，避免 label 数量不匹配报错
ax_ic.bar(range(n_top), top_10_ic_values, color='green', label='Top 10 IC')
ax_ic.bar(range(n_top, n_top + n_bottom), bottom_10_ic_values, color='red', label='Bottom 10 IC')

ax_ic.set_xticks(range(len(all_ic_names)))
ax_ic.set_xticklabels(all_ic_names, rotation=90, ha='right', fontsize=9)
ax_ic.set_xlabel('Features', fontsize=11)
ax_ic.set_ylabel('Information Coefficient (IC)', fontsize=11)
ax_ic.set_title(f'Top/Bottom 10 Features by IC | Symbol: {symbol} | Mode: {mode}| {operation}', fontsize=13, fontweight='bold')
ax_ic.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax_ic.grid(True, axis='y', linestyle=':', alpha=0.7)
ax_ic.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'output/ic_ranking_{symbol}_{label_name}_delay{delay_precentile}_{operation}_mode{mode}.png', dpi=300, bbox_inches='tight')
plt.close(fig_ic)
print(f"✓ Saved IC plot → ic_ranking_*.png")

# -----------------------------------------------------------------------------
# 2️⃣ IR Plot: Top/Bottom 10 Features by Information Ratio
# -----------------------------------------------------------------------------
fig_ir, ax_ir = plt.subplots(figsize=(14, 8))
all_ir_names = top_10_ir_feature_names + bottom_10_ir_feature_names
n_top_ir = len(top_10_ir_feature_names)
n_bottom_ir = len(bottom_10_ir_feature_names)

ax_ir.bar(range(n_top_ir), top_10_ir_values, color='blue', label='Top 10 IR')
ax_ir.bar(range(n_top_ir, n_top_ir + n_bottom_ir), bottom_10_ir_values, color='orange', label='Bottom 10 IR')

ax_ir.set_xticks(range(len(all_ir_names)))
ax_ir.set_xticklabels(all_ir_names, rotation=90, ha='right', fontsize=9)
ax_ir.set_xlabel('Features', fontsize=11)
ax_ir.set_ylabel('Information Ratio (IR)', fontsize=11)
ax_ir.set_title(f'Top/Bottom 10 Features by IR | Symbol: {symbol} | Mode: {mode}| {operation}', fontsize=13, fontweight='bold')
ax_ir.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax_ir.grid(True, axis='y', linestyle=':', alpha=0.7)
ax_ir.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'output/ir_ranking_{symbol}_{label_name}_delay{delay_precentile}_{operation}_mode{mode}.png', dpi=300, bbox_inches='tight')
plt.close(fig_ir)
print(f"✓ Saved IR plot → ir_ranking_*.png")

# -----------------------------------------------------------------------------
# 3️⃣ Absolute Difference Plot: 10 Features with Smallest |Predicted - Actual|
# -----------------------------------------------------------------------------
# (注：此处假设 top_10_ab_dif_feature_names / values 已在上方正确计算)
fig_ad, ax_ad = plt.subplots(figsize=(14, 8))
ax_ad.bar(range(len(top_10_ab_dif_feature_names)), top_10_ab_dif_values, color='cyan', edgecolor='darkcyan', label='Top 10 Smallest AD')
ax_ad.set_xticks(range(len(top_10_ab_dif_feature_names)))
ax_ad.set_xticklabels(top_10_ab_dif_feature_names, rotation=90, ha='right', fontsize=9)
ax_ad.set_xlabel('Features', fontsize=11)
ax_ad.set_ylabel('Mean Absolute Difference (MAD)', fontsize=11)
ax_ad.set_title(f'10 Features with Smallest Prediction Error | Symbol: {symbol} | Mode: {mode}| {operation}', fontsize=13, fontweight='bold')
ax_ad.grid(True, axis='y', linestyle=':', alpha=0.7)
ax_ad.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'output/ae_ranking_{symbol}_{label_name}_delay{delay_precentile}_{operation}_mode{mode}.png', dpi=300, bbox_inches='tight')
plt.close(fig_ad)
print(f"✓ Saved AE plot → ae_ranking_*.png")

df['const.'] = 1.0
feature_cols = selected_feature_cols + [f'BWT_{w}' for w in weights]
feature_cols.append('const.')

print(f"Selected {len(feature_cols)} features")

# -----------------------------------------------------------------------------
# 4 Full IC Plot: All Features by Information Coefficient
# -----------------------------------------------------------------------------
fig_ic, ax_ic = plt.subplots(figsize=(14, 8))
all_ic_names = top_all_feature_names
n_top = len(top_all_feature_names)

# ✅ 核心修复：分开绘制，避免 label 数量不匹配报错
ax_ic.bar(range(n_top), top_all_ic_values, color='green', label='Sorted IC')

    # ax_ic.set_xticks(range(len(all_ic_names)))
    # ax_ic.set_xticklabels(all_ic_names, rotation=90, ha='right', fontsize=9)
ax_ic.set_xlabel('Features', fontsize=11)
ax_ic.set_ylabel('Information Coefficient (IC)', fontsize=11)
ax_ic.set_title(f'Sorted Features by IC | Symbol: {symbol} | Mode: {mode}| {operation}', fontsize=13, fontweight='bold')
ax_ic.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax_ic.grid(True, axis='y', linestyle=':', alpha=0.7)
ax_ic.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'output/ic_ranking_all_{symbol}_{label_name}_delay{delay_precentile}_{operation}_mode{mode}.png', dpi=300, bbox_inches='tight')
plt.close(fig_ic)
print(f"✓ Saved All IC plot → ic_ranking_all_*.png")


fig_scatt, axes_scatt = plt.subplots(3,2, figsize=(18, 16), sharex=True)

all_ic_names = top_10_feature_names + bottom_10_feature_names
# top1_feature_names = top_10_feature_names[0]
# bottom_1_feature_name = bottom_10_feature_names[0]
# valid_idx = df[[top_10_feature_names[:3], bottom_10_feature_names[:3], label_name]].dropna().index
valid_idx = df[
    top_10_feature_names[:3] + bottom_10_feature_names[:3] + [label_name]
].dropna().index
if len(valid_idx) < 3:  # 样本过少时 spearmanr 无统计意义
    print('No enough data for scatter plots')
for i in range(3):
    axes_scatt[i, 0].set_title(f'Top {i+1} | Symbol: {symbol} | Mode: {mode}| {operation} |IC = {res_ic_ir_single[ic_ir_list_single.index(top_10_feature_names[i]), 0]:.4f}', fontsize=13, fontweight='bold')
    axes_scatt[i, 1].set_title(f'Bottom {i+1} | Symbol: {symbol} | Mode: {mode}| {operation} |IC = {res_ic_ir_single[ic_ir_list_single.index(bottom_10_feature_names[i]), 0]:.4f}', fontsize=13, fontweight='bold')
    axes_scatt[i, 0].set_xlabel(f'Feature Value ({top_10_feature_names[i]})', fontsize=11)
    axes_scatt[i, 0].set_ylabel(f'Label Value ({label_name})', fontsize=11)
    axes_scatt[i, 1].set_xlabel(f'Feature Value ({bottom_10_feature_names[i]})', fontsize=11)
    axes_scatt[i, 1].set_ylabel(f'Label Value ({label_name})', fontsize=11)
    x1 = df.loc[valid_idx, top_10_feature_names[i]]
    x2 = df.loc[valid_idx, bottom_10_feature_names[i]]
    y = df.loc[valid_idx, label_name]
    axes_scatt[i, 0].scatter(x1, y, alpha=0.5, label=f'Top {i+1} vs y')
    axes_scatt[i, 0].grid(True)
    axes_scatt[i, 0].legend()
    axes_scatt[i, 1].scatter(x2, y, alpha=0.5, label=f'Bottom {i+1} vs y')
    axes_scatt[i, 1].grid(True)
    axes_scatt[i, 1].legend()
# axes_scatt[0].scatter(x1, y, alpha=0.5, label='Top 1 vs y')
# axes_scatt[0].set_xlabel('Feature Value (Top 1)')
# axes_scatt[0].set_ylabel('Label value')
# axes_scatt[0].set_title(f'Top 1 vs Label ({top_10_feature_names[0]}, {label_name}| {operation}| {mode}|IC ={res_ic_ir_single[ic_ir_list_single.index(top_10_feature_names[0]), 0]:.4f})')
# axes_scatt[0].grid(True)
# axes_scatt[0].legend()
# axes_scatt[1].scatter(x2, y, alpha=0.5, label='Bottom 1 vs y')
# axes_scatt[1].set_xlabel('Feature Value (Bottom 1)')
# axes_scatt[1].set_ylabel('Label value')
# axes_scatt[1].set_title(f'Bott. 1  vs Label ({bottom_10_feature_names[0]}, {label_name}| {operation}| {mode}|IC ={res_ic_ir_single[ic_ir_list_single.index(bottom_10_feature_names[0]), 0]:.4f})')
# axes_scatt[1].grid(True)
# axes_scatt[1].legend()
# axes_scatt[2].scatter(x1, x2, alpha=0.5, label='Top 1 vs Bottom 1')
# axes_scatt[2].set_xlabel('Feature Value (Top 1)')
# axes_scatt[2].set_ylabel('Feature Value (Bottom 1)')
# axes_scatt[2].set_title(f'Top 1 vs Bott. 1 ({top1_feature_name}, {bottom_1_feature_name}| {operation}| {mode})')
# axes_scatt[2].grid(True)
# axes_scatt[2].legend()

plt.tight_layout()
plt.savefig(f'output/scatter_{symbol}_{label_name}_delay{delay_precentile}_{operation}_mode{mode}.png', dpi=300, bbox_inches='tight')
plt.close(fig_scatt)
# plt.show()


fig_scatt2, axes_scatt2 = plt.subplots(4,3, figsize=(18, 16), sharex=True)

feature_names_origin = ['BWT_baa', 'BWT_bab', 'BWT_bba', 'BWT_bbb']
valid_idx = df[['BWT_baa_0', 'BWT_bab_0', 'BWT_bba_0', 'BWT_bbb_0', label_name]].dropna().index
if len(valid_idx) < 3:  # 样本过少时 spearmanr 无统计意义
    print('No enough data for scatter plots')
for i, feature_name in enumerate(feature_names_origin):
    x1 = df.loc[valid_idx, f"{feature_name}_-5"]
    x2 = df.loc[valid_idx, f"{feature_name}_0"]
    x3 = df.loc[valid_idx, f"{feature_name}_5"]
    y = df.loc[valid_idx, label_name]
    axes_scatt2[i, 0].scatter(x1, y, alpha=0.5, label=f'{feature_name} vs y')
    axes_scatt2[i, 0].set_xlabel(f'Feature Value ({feature_name}_-5)')
    axes_scatt2[i, 0].set_ylabel('Label value')
    axes_scatt2[i, 0].set_title(f'{feature_name} vs Label ({feature_name}, {label_name}| {operation}| {mode})')
    axes_scatt2[i, 0].grid(True)
    axes_scatt2[i, 0].legend()

    axes_scatt2[i, 1].scatter(x2, y, alpha=0.5, label=f'{feature_name} vs y')
    axes_scatt2[i, 1].set_xlabel(f'Feature Value ({feature_name}_0)')
    axes_scatt2[i, 1].set_ylabel('Label value')
    axes_scatt2[i, 1].set_title(f'{feature_name} vs Label ({feature_name}, {label_name}| {operation}| {mode})')
    axes_scatt2[i, 1].grid(True)
    axes_scatt2[i, 1].legend()

    axes_scatt2[i, 2].scatter(x3, y, alpha=0.5, label=f'{feature_name} vs y')
    axes_scatt2[i, 2].set_xlabel(f'Feature Value ({feature_name}_5)')
    axes_scatt2[i, 2].set_ylabel('Label value')
    axes_scatt2[i, 2].set_title(f'{feature_name} vs Label ({feature_name}, {label_name}| {operation}| {mode})')
    axes_scatt2[i, 2].grid(True)
    axes_scatt2[i, 2].legend()

plt.tight_layout()
plt.savefig(f'output/scatter2_{symbol}_{label_name}_delay{delay_precentile}_{operation}_mode{mode}.png', dpi=300, bbox_inches='tight')
plt.close(fig_scatt2)

# Prepare data for modeling
df_sample = df.copy()
X = df_sample[feature_cols].values
print(f"Feature matrix shape: {X.shape}")
print(df_sample[feature_cols].info())

# Generate labels based on the selected label column
Y = df_sample[[label_name, 'gain_vs_threshold']].values
# y = np.squeeze(y)  # Convert to 1D array if it's a single column# Fix: Remove redundant train_test_split and use consistent test_size
test_size = 0.2  # Changed from 0.1 to 0.2 for better evaluation
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Removed redundant line
X_train, X_test = X[:int(len(X) * (1 - test_size))], X[int(len(X) * (1 - test_size)):]
Y_train, Y_test = Y[:int(len(Y) * (1 - test_size))], Y[int(len(Y) * (1 - test_size)):]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

# x_thres = X_test[:, feature_cols.index('threshold')]
x_thres = X_test[:, feature_cols.index('const.')]
# Normalize features and labels
if normalize_X == 1:
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    Y_train_mean = Y_train.mean(axis=0)
    Y_train_std = Y_train.std(axis=0)
    y_train_mean = Y_train[:, 0].mean()
    y_train_std = Y_train[:, 0].std()
else:
    X_train_mean = np.zeros(X_train.shape[1])
    X_train_std = np.ones(X_train.shape[1])
    Y_train_mean = np.zeros(Y_train.shape[1])
    Y_train_std = np.ones(Y_train.shape[1])
    y_train_mean = 0
    y_train_std = 1

X_train = (X_train - X_train_mean) / (X_train_std + TOLERENCE)
Y_train = (Y_train - Y_train_mean) / (Y_train_std + TOLERENCE)
X_test = (X_test - X_train_mean) / (X_train_std + TOLERENCE)
# Note: Y_test is kept original for final metric evaluation, but needs normalization for CNN importance scoring
y_test = Y_test[:, 0]  # Assuming the first column is the main label for evaluation
y_train = Y_train[:, 0]  # Assuming the first column is the main label
label_test = Y_test[:, 1]  # gain_vs_threshold for label prediction plot
label_train = Y_train[:, 1]  # gain_vs_threshold for label prediction plot


