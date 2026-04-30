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
mode = 0
delay_exec = ''
normalize_X = 0
# operation = 'open2'
operation = 'close2'
delay_precentile = 80
beta = 1
symbol = 'all'
# symbol = 'ZENUSDT'

######### label
label_name = 'gain_vs_threshold' 

# Define the folder path
folder_path = f'dataset/preprocessed{delay_exec}/mode{mode}/'


# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Read and combine all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)
print(f"Combined {len(csv_files)} files")
print(f"Total rows: {len(combined_df)}")
print(combined_df.info(verbose=True, show_counts=True))
print(f"List of feature columns: {combined_df.columns.tolist()}")

selected_cols = ['gain_vs_threshold', 'basis_slippage', 'symbol', 'trade_mode', 'operation', 
                 'exec_ts_utc', 'window_start_ms', 'window_end_ms', 'execute_delay_ms', 'timer_start_ts', 'taker_exec_ts', 
                 'threshold', 'basis_expected', 'basis_executed', 
                 'basis_ask_mean', 'basis_bid_mean', 'basis_ask_std', 'basis_bid_std', 
                 'basis_ask_open', 'basis_bid_open', 'basis_ask_close', 'basis_bid_close', 
                 'basis_ask_high', 'basis_bid_high', 'basis_ask_low', 'basis_bid_low', 
                 'basis_ask_adjusted_mean', 'basis_bid_adjusted_mean', 'basis_ask_adjusted_std', 'basis_bid_adjusted_std', 
                 'spot_spread_ticks', 'spot_spread_mean', 'swap_spread_ticks', 'swap_spread_mean', 
                 'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean', 'spot_depth5_imbalance_mean', 'swap_depth5_imbalance_mean', 
                 'spot_depth1_bid_ticks', 'spot_depth1_ask_ticks', 'swap_depth1_bid_ticks', 'swap_depth1_ask_ticks', 
                 'spot_volatility_ticks', 'swap_volatility_ticks', 'spot_price_return_2min', 'swap_price_return_2min', 
                 'spot_trade_volume_2min', 'spot_trade_count_2min', 
                 'spot_bid_price_mean', 'spot_ask_price_mean', 'swap_bid_price_mean', 'swap_ask_price_mean', 
                 'swap_trade_volume_2min', 'swap_trade_count_2min', 'spot_buy_trade_ratio', 
                 'basis_slippage_ticks', 'spot_midprice_mean', 'spot_midprice_std', 'spread_ticks', 
                 'depth_imbalance_mean', 'volatility_ticks', 'trade_volume_2min', 
                 'spot_midprice_open', 'spot_midprice_close', 'spot_midprice_high', 'spot_midprice_low', 
                 'swap_midprice_mean', 'swap_midprice_std', 'swap_buy_trade_ratio', 
                 'basis_mid_adjusted_mean', 'basis_mid_adjusted_std', 
                 'basis_mid_capped_mean', 'basis_mid_capped_std', 
                 'n_basis_bid_mix', 'n_basis_ask_mix', 'n_basis_capped', 
                 'basis_mid_weighted_mean_-1e6', 'basis_mid_weighted_mean_-1e5', 
                 'basis_mid_weighted_mean_-1e4', 'basis_mid_weighted_mean_-1e3', 
                 'basis_mid_weighted_mean_-1e2', 'basis_mid_weighted_mean_-1e1', 
                 'basis_mid_weighted_mean_-1e0', 'basis_mid_weighted_mean_0', 
                 'basis_mid_weighted_mean_1e0', 'basis_mid_weighted_mean_1e1', 
                 'basis_mid_weighted_mean_1e2', 'basis_mid_weighted_mean_1e3', 
                 'basis_mid_weighted_mean_1e4', 'basis_mid_weighted_mean_1e5', 'basis_mid_weighted_mean_1e6', 
                 'basis_mid_weighted_mean_-1e6_0.5', 'basis_mid_weighted_mean_-1e6_0.75', 
                 'basis_mid_weighted_mean_-1e6_1.0', 'basis_mid_weighted_mean_-1e5_0.5', 
                 'basis_mid_weighted_mean_-1e5_0.75', 'basis_mid_weighted_mean_-1e5_1.0', 
                 'basis_mid_weighted_mean_-1e4_0.5', 'basis_mid_weighted_mean_-1e4_0.75', 
                 'basis_mid_weighted_mean_-1e4_1.0', 'basis_mid_weighted_mean_-1e3_0.5', 
                 'basis_mid_weighted_mean_-1e3_0.75', 'basis_mid_weighted_mean_-1e3_1.0', 
                 'basis_mid_weighted_mean_-1e2_0.5', 'basis_mid_weighted_mean_-1e2_0.75', 
                 'basis_mid_weighted_mean_-1e2_1.0', 'basis_mid_weighted_mean_-1e1_0.5', 
                 'basis_mid_weighted_mean_-1e1_0.75', 'basis_mid_weighted_mean_-1e1_1.0', 
                 'basis_mid_weighted_mean_-1e0_0.5', 'basis_mid_weighted_mean_-1e0_0.75', 
                 'basis_mid_weighted_mean_-1e0_1.0', 'basis_mid_weighted_mean_0_0.5', 
                 'basis_mid_weighted_mean_0_0.75', 'basis_mid_weighted_mean_0_1.0', 
                 'basis_mid_weighted_mean_1e0_0.5', 'basis_mid_weighted_mean_1e0_0.75', 
                 'basis_mid_weighted_mean_1e0_1.0', 'basis_mid_weighted_mean_1e1_0.5', 
                 'basis_mid_weighted_mean_1e1_0.75', 'basis_mid_weighted_mean_1e1_1.0', 
                 'basis_mid_weighted_mean_1e2_0.5', 'basis_mid_weighted_mean_1e2_0.75', 
                 'basis_mid_weighted_mean_1e2_1.0', 'basis_mid_weighted_mean_1e3_0.5', 
                 'basis_mid_weighted_mean_1e3_0.75', 'basis_mid_weighted_mean_1e3_1.0', 
                 'basis_mid_weighted_mean_1e4_0.5', 'basis_mid_weighted_mean_1e4_0.75', 
                 'basis_mid_weighted_mean_1e4_1.0', 'basis_mid_weighted_mean_1e5_0.5', 
                 'basis_mid_weighted_mean_1e5_0.75', 'basis_mid_weighted_mean_1e5_1.0', 
                 'basis_mid_weighted_mean_1e6_0.5', 'basis_mid_weighted_mean_1e6_0.75', 'basis_mid_weighted_mean_1e6_1.0'
                 ]

weighted_feature_cols = [col for col in selected_cols if 'basis_mid_weighted_mean' in col]
# Obtain weights from the column names
weights = []
for col in weighted_feature_cols:
    weight_str = col.replace('basis_mid_weighted_mean_', '')
    weights.append(weight_str)

print(f"Extracted weights: {weights}")


# Sort by exec_ts_utc
# combined_df['exec_ts_utc'] = pd.to_datetime(combined_df['exec_ts_utc'])
combined_df['exec_ts_utc'] = pd.to_datetime(
    combined_df['exec_ts_utc'],
    format='ISO8601',
    utc=True
)
combined_df = combined_df.sort_values('exec_ts_utc')

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

# 基础衍生特征
# new_cols_dict['basis_slippage_rate'] = (df['basis_executed'] - df['basis_expected']) / (df['basis_expected'] + 1e-12)
# new_cols_dict['basis_ask_k_volatility'] = (df['basis_ask_high'] - df['basis_ask_low']) / (np.abs(df['basis_ask_close'] - df['basis_ask_open']) + 1e-12)
# new_cols_dict['basis_bid_k_volatility'] = (df['basis_bid_high'] - df['basis_bid_low']) / (np.abs(df['basis_bid_close'] - df['basis_bid_open']) + 1e-12)

if 'basis_ask_mean' in df.columns and 'basis_bid_mean' in df.columns:
    new_cols_dict['basis_mid_mean'] = (df['basis_ask_mean'] + df['basis_bid_mean']) / 2
    new_cols_dict['basis_mid_to_thres'] = new_cols_dict['basis_mid_mean'] - df['threshold']
    feature_cols.append('basis_mid_to_thres')

# if 'basis_ask_adjusted_mean' in df.columns and 'basis_bid_adjusted_mean' in df.columns:
#     new_cols_dict['basis_mid_adjusted_mean'] = (df['basis_ask_adjusted_mean'] + df['basis_bid_adjusted_mean']) / 2
#     new_cols_dict['basis_adjusted_mid_to_thres'] = new_cols_dict['basis_mid_adjusted_mean'] - df['threshold']
#     feature_cols.append('basis_adjusted_mid_to_thres')

new_cols_dict['cap_to_thres'] = df['basis_mid_capped_mean'] - df['threshold']
feature_cols.append('cap_to_thres')
# 安全提取加权特征（自动跳过不存在的列，防止 KeyError）
valid_weights = []
for w_str in weights:  # weights 当前为字符串列表，如 '-6.0', 'zero'
    src_col = f'basis_mid_weighted_mean_{w_str}'
    if src_col in df.columns:
        new_cols_dict[f'BWT_{w_str}'] = df[src_col] - df['threshold']
        feature_cols.append(f'BWT_{w_str}')
        # selected_feature_cols.append(src_col)
        valid_weights.append(w_str)
    else:
        print(f"⚠️ 跳过缺失列: {src_col}")

# 更新 weights 仅保留有效列，保证后续 IC 计算与绘图严格对齐
weights = valid_weights

# 一次性追加所有新列到 DataFrame
if new_cols_dict:
    df = pd.concat([df, pd.DataFrame(new_cols_dict, index=df.index)], axis=1)

# print(filtered_df['exec_ts_utc'])
# df['date'] = df['exec_ts_utc'].dt.date
# ✅ 替换原日期创建
# 🛡️ 1. 消除 PerformanceWarning：在重操作前强制内存重排（defragment）
df = df.copy()

# 2. 创建日期列（使用 floor 保持 datetime64 类型）
df['date'] = df['exec_ts_utc'].dt.floor('D')

# Compute IC/IR for feature_cols
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
        return stats.spearmanr(x, y)[0]
    daily_ic = df.groupby('date').apply(calc_daily_ic)
    daily_ic = pd.to_numeric(daily_ic, errors='coerce')
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    ir = ic_mean / ic_std if pd.notna(ic_std) and ic_std > 0 else np.nan
    print(f"✅ IC of {col}: {ic_mean:.8f} (±{ic_std:.8f}) | IR: {ir:.8f}")

# if 'cap_to_thres' in df.columns:
#     def calc_daily_ic_cap(group):
#         valid_idx = group[['cap_to_thres', label_name]].dropna().index
#         if len(valid_idx) < 3:
#             return np.nan
#         x = group.loc[valid_idx, 'cap_to_thres']
#         y = group.loc[valid_idx, label_name]
#         return stats.spearmanr(x, y)[0]

#     daily_ic_cap = df.groupby('date').apply(calc_daily_ic_cap)
#     daily_ic_cap = pd.to_numeric(daily_ic_cap, errors='coerce')

#     ic_mean_cap = daily_ic_cap.mean()
#     ic_std_cap = daily_ic_cap.std()
#     ir_cap = ic_mean_cap / ic_std_cap if pd.notna(ic_std_cap) and ic_std_cap > 0 else np.nan

#     print(f"✅ IC of cap_to_thres: {ic_mean_cap:.8f} (±{ic_std_cap:.8f}) | IR: {ir_cap:.8f}")

# 3. 定义要计算 IC/IR 的因子列表
# ic_ir_list = [f'BWT_{w}' for w in weights]
ic_ir_list = feature_cols
ic_ir_list_single = [col for col in ic_ir_list if 'BWT_' in col and '_' not in col.split('BWT_')[-1]]
ic_ir_list_dual = [col for col in ic_ir_list if 'BWT_' in col and '_' in col.split('BWT_')[-1]]
res_ic_ir_single = np.zeros((len(ic_ir_list_single), 2))
res_ic_ir_dual = np.zeros((len(ic_ir_list_single), 3, 2))

print(f"ICIR list single: {ic_ir_list_single}")
print(f"ICIR list dual: {ic_ir_list_dual}")
# 如果还需要计算原始 mid 因子的 IC，可取消下一行注释
# ic_ir_list.append('basis_mid_mean')
# 计算IC， IR， single 和 dual_mixture 分开计算并存储结果
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
        return stats.spearmanr(x, y)[0]

    # 计算每日 IC
    daily_ic = df.groupby('date').apply(calc_daily_ic)
    daily_ic = pd.to_numeric(daily_ic, errors='coerce')  # 强制转 float 防 object 类型报错

    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    ir = ic_mean / ic_std if pd.notna(ic_std) and ic_std > 0 else np.nan

    # print(f"✅ IC of {factor_name}: {ic_mean:.8f} (±{ic_std:.8f}) | IR: {ir:.8f}")
    res_ic_ir_single[i, 0] = ic_mean
    res_ic_ir_single[i, 1] = ir

for i, factor_name in enumerate(ic_ir_list_dual):
    if factor_name not in df.columns:
        print(f"⚠️ 跳过缺失因子: {factor_name}")
        continue
    w, alpha = factor_name.split('BWT_')[-1].split('_')[0], factor_name.split('BWT_')[-1].split('_')[1]
    alpha_mp = {'0.5': 0, '0.75': 1, '1.0': 2}
    j = alpha_mp.get(alpha, None)
    w_mp = {'-1e6': 0, '-1e5': 1, '-1e4': 2, '-1e3': 3, '-1e2': 4, '-1e1': 5, '-1e0': 6, '0': 7, '1e0': 8, '1e1': 9, '1e2': 10, '1e3': 11, '1e4': 12, '1e5': 13, '1e6': 14}
    i = w_mp.get(w, None)
    # print(f"Processing dual mixture factor: {factor_name} with w={w}, alpha={alpha} mapped to indices i={i}, j={j}")
    # convert factor_name like 'basis_mid_weighted_mean_-1e6_0.5' to extract w and alpha
    # w, alpha = reconstruct_weight_from_suffix(factor_name.split('_')[-1])

    def calc_daily_ic(group):
        # 🛡️ 核心修复：基于共同非空索引对齐因子与标签，确保 len(x) == len(y)
        valid_idx = group[[factor_name, label_name]].dropna().index
        if len(valid_idx) < 3:  # 样本过少时 spearmanr 无统计意义
            return np.nan
        x = group.loc[valid_idx, factor_name]
        y = group.loc[valid_idx, label_name]
        return stats.spearmanr(x, y)[0]

    # 计算每日 IC
    daily_ic = df.groupby('date').apply(calc_daily_ic)
    daily_ic = pd.to_numeric(daily_ic, errors='coerce')  # 强制转 float 防 object 类型报错

    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    ir = ic_mean / ic_std if pd.notna(ic_std) and ic_std > 0 else np.nan

    # print(f"✅ IC of {factor_name}: {ic_mean:.8f} (±{ic_std:.8f}) | IR: {ir:.8f}")
    # print(f"Storing results at res_ic_ir_dual[{i}, {j}] for factor {factor_name} with w={w} and alpha={alpha}")
    res_ic_ir_dual[i, j, 0] = ic_mean
    res_ic_ir_dual[i, j, 1] = ir
    # print(res_ic_ir_dual[:, :, 0])

# ✅ 替换为：安全提取数据并修复未定义变量
# ✅ 替换原绘图代码：安全提取有效数据，并将字符串权重转为浮点数用于坐标轴
# ✅ 新增：健壮的权重字符串解析函数
# ✅ 替换原 parse_weight_str 函数

from typing import Union, Tuple, List
import re
def reconstruct_weight_from_suffix(key: str) -> Union[float, Tuple[float, float]]:
    """
    逆向解析 single/dual_mixture_weighted_stats 生成的 key_suffix，复原对应的权重 w 及 alpha。
    
    参数:
        key: 格式如 '1e2', '-1e-3', '0', 或 '1e-1_0.5', '-1e2_1.0'
        
    返回:
        float: 仅含 w 时返回权重
        Tuple[float, float]: 含 alpha 时返回 (w, alpha)
    """
    key = key.strip()
    
    # 1. 分离 dual_mixture 的 alpha 部分
    if '_' in key:
        w_part, a_part = key.rsplit('_', 1)
        alpha = float(a_part)
    else:
        w_part = key
        alpha = None
        
    # 2. 解析 w 部分
    if w_part == '0':
        w = 0.0
    else:
        # 严格匹配 (-)1e(+/-)int 格式
        match = re.match(r'^(-?)1e(-?\d+)$', w_part)
        if not match:
            raise ValueError(f"⚠️ 无法解析权重后缀格式: '{w_part}'")
            
        sign_str, exp_str = match.groups()
        w = (-1.0 if sign_str else 1.0) * (10 ** int(exp_str))
        
    return w, alpha 


# ✅ 替换后续绘图坐标解析部分
plot_weights = [reconstruct_weight_from_suffix(w) for w in weights]
print(f"Parsed plot weights: {plot_weights}, length: {len(plot_weights)}")
N_plot = len(plot_weights)
half_N = N_plot

# 严格过滤 NaN 保证 x/y 长度一致
valid_mask = [pd.notna(x[0]) for x in res_ic_ir_single]
valid_weights = [w for w, v in zip(plot_weights, valid_mask) if v]
ic_values = [x[0] for x in res_ic_ir_single if pd.notna(x[0])]
ir_values = [x[1] for x in res_ic_ir_single if pd.notna(x[1])]

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
# plt.figure(figsize=(12, 10))
# plt.subplot(2, 1, 1)
if len(ic_ir_list_single) > 0:
    axes[0].scatter(range(len(ic_ir_list_single)), ic_values, label='IC: Weighted Basis Mid')
    # plt.xscale('symlog')  # 使用对称 log 坐标轴以更好地展示正负权重的 IC 值
axes[0].set_xticks(range(len(ic_ir_list_single)))
axes[0].set_xticklabels([ic_ir_list_single[j] for j in range(len(ic_ir_list_single))], rotation=90)
axes[0].set_xlabel('Features')
axes[0].set_ylabel('IC')
axes[0].set_title('Daily Cross-Sectional IC')
axes[0].grid(True); axes[0].legend()

# plt.subplot(2, 1, 2)
if len(ic_ir_list_single) > 0:
    axes[1].scatter(range(len(ic_ir_list_single)), ir_values, label='IR: Weighted Basis Mid', color='orange')
    # plt.xscale('symlog')  # 使用对称 log 坐标轴以更好地展示正负权重的 IR 值
axes[1].set_xticks(range(len(ic_ir_list_single)))
axes[1].set_xticklabels([ic_ir_list_single[j] for j in range(len(ic_ir_list_single))], rotation=90)
axes[1].set_xlabel('Features')
axes[1].set_ylabel('IR')
axes[1].set_title('Information Ratio (IR)')
axes[1].grid(True); axes[1].legend()

# Plot heatmap for dual mixture IC values
if len(ic_ir_list_dual) > 0:
    # print(res_ic_ir_dual[:, :, 0])
    max_ic_dual = np.nanmax(res_ic_ir_dual[:, :, 0])
    min_ic_dual = np.nanmin(res_ic_ir_dual[:, :, 0])
    axes[2].imshow(res_ic_ir_dual[:, :, 0], aspect='auto', cmap='viridis', vmin=min_ic_dual, vmax=max_ic_dual)
    # axes[2].colorbar()
    axes[2].set_xticks(range(res_ic_ir_dual.shape[1]))
    axes[2].set_xticklabels(['alpha=0.5', 'alpha=0.75', 'alpha=1.0'], rotation=90)
    axes[2].set_yticks(range(res_ic_ir_dual.shape[0]))
    axes[2].set_yticklabels([ic_ir_list_single[j].split('BWT_')[-1] for j in range(len(ic_ir_list_single))])
    axes[2].set_xlabel('Alpha')
    axes[2].set_ylabel('Weight')
    axes[2].set_title('Daily Cross-Sectional IC for Dual Mixture')

if len(ic_ir_list_dual) > 0:
    max_ir_dual = np.nanmax(res_ic_ir_dual[:, :, 1])
    min_ir_dual = np.nanmin(res_ic_ir_dual[:, :, 1])
    axes[3].imshow(res_ic_ir_dual[:, :, 1], aspect='auto', cmap='viridis', vmin=min_ir_dual, vmax=max_ir_dual)
    # axes[3].colorbar()
    axes[3].set_xticks(range(res_ic_ir_dual.shape[1]))
    axes[3].set_xticklabels(['alpha=0.5', 'alpha=0.75', 'alpha=1.0'], rotation=90)
    axes[3].set_yticks(range(res_ic_ir_dual.shape[0]))
    axes[3].set_yticklabels([ic_ir_list_single[j].split('BWT_')[-1] for j in range(len(ic_ir_list_single))])
    axes[3].set_xlabel('Alpha')
    axes[3].set_ylabel('Weight')
    axes[3].set_title('Information Ratio (IR) for Dual Mixture')

fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min_ic_dual, vmax=max_ic_dual)), ax=axes[2], orientation='vertical', label='IC Value')
fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min_ir_dual, vmax=max_ir_dual)), ax=axes[3], orientation='vertical', label='IR Value')
plt.tight_layout()
os.makedirs('output', exist_ok=True)
plt.savefig(f'output/ic_ir_{symbol}_{label_name}_nMod{len(models)}delay{delay_precentile}{operation}_nml{normalize_X}_mode{mode}.png', bbox_inches='tight')
plt.show()

df['const.'] = 1.0
feature_cols = selected_feature_cols + [f'BWT_{w}' for w in weights]
feature_cols.append('const.')

print(f"Selected {len(feature_cols)} features")

# Visualize correlation
label_names = ['gain_vs_threshold']
df_corr = df[feature_cols + label_names]#.corr()
corr_matrix = df_corr.corr()


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


# Ensure X_train/X_test are pure numpy arrays to avoid LightGBM feature name warnings
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

# if 'OLS Regression' in models:
#     results['OLS Regression'] = {}
#     # Train an OLS regression model using statsmodels
#     # X_train_sm = sm.add_constant(X_train)  # Add intercept term
#     X_train_sm = X_train.copy()  # Ensure it's a pure numpy array
#     model_ols = sm.OLS(y_train, X_train_sm).fit()
#     # X_test_sm = sm.add_constant(X_test)
#     X_test_sm = X_test.copy()  # Ensure it's a pure numpy array
#     y_pred_ols_norm = model_ols.predict(X_test_sm)
#     y_pred_ols = y_pred_ols_norm * y_train_std + y_train_mean# Denormalize predictions
#     y_pred_ols_train_norm = model_ols.predict(X_train_sm)
#     y_pred_ols_train = y_pred_ols_train_norm * y_train_std + y_train_mean  # Denormalize train predictions
#     label_pred = y_pred_ols - beta * x_thres  

#     print(model_ols.summary())
#     # importance_ols = model_ols.params[0:]  # Exclude intercept
#     importance_ols = model_ols.tvalues[0:]
#     indices_ols = np.argsort(np.abs(importance_ols))[::-1]
#     print(f"y_pred_ols: {y_pred_ols}")
#     mse_ols = mean_squared_error(y_test, y_pred_ols)
#     r2_ols = r2_score(y_test, y_pred_ols)
#     print(f"OLS Regression - MSE: {mse_ols:.4f}, R2: {r2_ols:.4f}")
#     results['OLS Regression']['MSE'] = mse_ols
#     results['OLS Regression']['R2'] = r2_ols
#     results['OLS Regression']['y_pred'] = y_pred_ols
#     results['OLS Regression']['y_pred_train'] = y_pred_ols_train
#     results['OLS Regression']['importance'] = importance_ols
#     results['OLS Regression']['indices'] = indices_ols
#     results['OLS Regression']['label_pred'] = label_pred


# if 'Linear Regression' in models:
#     results['Linear Regression'] = {}
#     # Train a linear regression model
#     model_LR = LinearRegression()
#     model_LR.fit(X_train, y_train)
#     y_pred_LR = model_LR.predict(X_test)
#     y_pred_LR = y_pred_LR * y_train_std + y_train_mean# Denormalize predictions
#     y_pred_LR_train = model_LR.predict(X_train) * y_train_std + y_train_mean  # Denormalize train predictions
#     label_pred_LR = y_pred_LR - beta * x_thres
#     mse_LR = mean_squared_error(y_test, y_pred_LR)
#     r2_LR = r2_score(y_test, y_pred_LR)
#     print(f"Linear Regression - MSE: {mse_LR:.4f}, R2: {r2_LR:.4f}")

#     # Feature importance for linear regression (absolute value of coefficients)
#     print(model_LR.coef_.shape)
#     coef_importance = model_LR.coef_
#     indices_lr = np.argsort(np.abs(coef_importance))[::-1]
#     results['Linear Regression']['MSE'] = mse_LR
#     results['Linear Regression']['R2'] = r2_LR
#     results['Linear Regression']['y_pred'] = y_pred_LR
#     results['Linear Regression']['y_pred_train'] = y_pred_LR_train
#     results['Linear Regression']['importance'] = coef_importance
#     results['Linear Regression']['indices'] = indices_lr
#     results['Linear Regression']['label_pred'] = label_pred_LR


# if 'LightGBM Regressor' in models:
#     # Train a lightGBM model
#     model_lgb = lgb.LGBMRegressor(n_estimators=10000, reg_alpha=0.5, 
#                                   max_depth=20, random_state=42,
#                                   verbosity = -1)
#     model_lgb.fit(X_train, y_train)
#     y_pred_lgb = model_lgb.predict(X_test)
#     y_pred_lgb = y_pred_lgb * y_train_std + y_train_mean  # Denormalize predictions
#     y_pred_lgb_train = model_lgb.predict(X_train) * y_train_std + y_train_mean  # Denormalize train predictions
#     label_pred_lgb = y_pred_lgb - beta * x_thres
#     mse_lgb = mean_squared_error(y_test, y_pred_lgb)
#     r2_lgb = r2_score(y_test, y_pred_lgb)
#     lgb_importances = model_lgb.feature_importances_
#     indices_lgb = np.argsort(lgb_importances)[::-1]
#     print(f"LightGBM Regressor - MSE: {mse_lgb:.4f}, R2: {r2_lgb:.4f}")
#     results['LightGBM Regressor'] = {}
#     results['LightGBM Regressor']['MSE'] = mse_lgb
#     results['LightGBM Regressor']['R2'] = r2_lgb
#     results['LightGBM Regressor']['y_pred'] = y_pred_lgb
#     results['LightGBM Regressor']['y_pred_train'] = y_pred_lgb_train
#     results['LightGBM Regressor']['importance'] = lgb_importances
#     results['LightGBM Regressor']['indices'] = indices_lgb
#     results['LightGBM Regressor']['label_pred'] = label_pred_lgb


# if 'XGBoost Regressor' in models:
#     results['XGBoost Regressor'] = {}
#     # Train a XGBoost model
#     model_xgb = xgb.XGBRegressor(n_estimators=20000, max_depth=10,
#                                  device='cuda',
#                                  random_state=42)
#     model_xgb.fit(X_train, y_train)
#     y_pred_xgb = model_xgb.predict(X_test)
#     y_pred_xgb = y_pred_xgb * y_train_std + y_train_mean  # Denormalize predictions
#     y_pred_xgb_train = model_xgb.predict(X_train) * y_train_std + y_train_mean  # Denormalize train predictions
#     label_pred_xgb = y_pred_xgb - beta * x_thres
#     mse_xgb = mean_squared_error(y_test, y_pred_xgb)
#     r2_xgb = r2_score(y_test, y_pred_xgb)
#     importance_xgb = model_xgb.feature_importances_
#     indices_xgb = np.argsort(importance_xgb)[::-1]
#     print(f"XGBoost Regressor - MSE: {mse_xgb:.4f}, R2: {r2_xgb:.4f}")
#     results['XGBoost Regressor']['MSE'] = mse_xgb
#     results['XGBoost Regressor']['R2'] = r2_xgb
#     results['XGBoost Regressor']['y_pred'] = y_pred_xgb
#     results['XGBoost Regressor']['y_pred_train'] = y_pred_xgb_train
#     results['XGBoost Regressor']['importance'] = importance_xgb
#     results['XGBoost Regressor']['indices'] = indices_xgb
#     results['XGBoost Regressor']['label_pred'] = label_pred_xgb

# if 'Random Forest Regressor' in models:
#     results['Random Forest Regressor'] = {}
#     # Train a Random Forest Regressor
#     model_rf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42)
#     model_rf.fit(X_train, y_train)
#     y_pred_rf = model_rf.predict(X_test)
#     y_pred_rf = y_pred_rf * y_train_std + y_train_mean  # Denormalize predictions
#     y_pred_rf_train = model_rf.predict(X_train) * y_train_std + y_train_mean  # Denormalize train predictions
#     label_pred_rf = y_pred_rf - beta * x_thres
#     mse_rf = mean_squared_error(y_test, y_pred_rf)
#     r2_rf = r2_score(y_test, y_pred_rf)
#     print(f"Random Forest Regressor - MSE: {mse_rf:.4f}, R2: {r2_rf:.4f}")

#     # Feature importance plot
#     importances = model_rf.feature_importances_
#     indices = np.argsort(importances)[::-1]
#     results['Random Forest Regressor']['MSE'] = mse_rf
#     results['Random Forest Regressor']['R2'] = r2_rf
#     results['Random Forest Regressor']['y_pred'] = y_pred_rf
#     results['Random Forest Regressor']['y_pred_train'] = y_pred_rf_train
#     results['Random Forest Regressor']['importance'] = importances
#     results['Random Forest Regressor']['indices'] = indices
#     results['Random Forest Regressor']['label_pred'] = label_pred_rf


# Plot feature importance for all models
# models_plot_imp = [model for model in models if 'importance' in results[model]]  # Exclude CNN from importance plot
# nplot_1 = len(models_plot_imp)
# fig, axes = plt.subplots(1, nplot_1, figsize=(24, 6))
# for i in range(nplot_1):
#     axes[i].grid(True)
#     axes[i].set_title(f"{models_plot_imp[i]} (sym: {symbol}, {label_name}),\n R2: {results[models_plot_imp[i]]['R2']:.4f}")
#     # print(f"X_train shape: {X_train.shape}, importance shape: {results[models_plot_imp[i]]['importance'].shape}")
#     axes[i].bar(range(X_train.shape[1]), results[models_plot_imp[i]]['importance'][results[models_plot_imp[i]]['indices']], align="center")
#     axes[i].set_xticks(range(X_train.shape[1]))
#     axes[i].set_xticklabels([feature_cols[j] for j in results[models_plot_imp[i]]['indices']], rotation=90)
#     axes[i].set_xlim([-1, X_train.shape[1]])
#     axes[i].set_ylabel("Importance")
#     axes[i].set_xlabel("Feature")

# plt.tight_layout()
# plt.savefig(f'output/ft_imp_{symbol}_{label_name}_nMod{len(models)}_{label_name}_nFts{X_train.shape[1]}{delay_exec}_delay{delay_precentile}_{operation}_nml{normalize_X}_mode{mode}.png', bbox_inches='tight')

# plt.show()


