import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

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
# delay_exec = ''
# delay_exec = '_2'
delay_exec = '_4'
normalize_X = 0
# operation = 'open2'
operation = 'close2'
delay_precentile = 50
beta = 1
symbol = 'all'
# symbol = 'ZENUSDT'

######### label
label_name = 'gain_vs_threshold' # basis_executed - threshold
# label_name = 'basis_expected_to_thres'
# label_name = 'basis_executed'
# label_name = 'basis_slippage'

# Define the folder path
folder_path = f'dataset/preprocessed{delay_exec}/mode{mode}/'
# folder_path = 'dataset/preprocessed/mode2/'


# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Read and combine all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)
print(f"Combined {len(csv_files)} files")
print(f"Total rows: {len(combined_df)}")
print(combined_df.info())
print(f"List of feature columns: {combined_df.columns.tolist()}")

selected_cols = ['gain_vs_threshold', 'basis_slippage', 'symbol', 'trade_mode', 'operation', 
                 'exec_ts_utc', 'window_start_ms', 'window_end_ms', 'execute_delay_ms', 'timer_start_ts', 'taker_exec_ts', 
                 'threshold', 'basis_expected', 'basis_executed', 
                 'basis_ask_mean', 'basis_bid_mean', 'basis_ask_std', 'basis_bid_std', 
                 'basis_ask_open', 'basis_bid_open', 'basis_ask_close', 'basis_bid_close', 'basis_ask_high', 'basis_bid_high', 
                 'basis_ask_low', 'basis_bid_low', 
                #  'basis_ask_adjusted_mean', 'basis_bid_adjusted_mean', 'basis_ask_adjusted_std', 'basis_bid_adjusted_std', 
                #  'spot_spread_ticks', 'spot_spread_mean', 'swap_spread_ticks', 'swap_spread_mean', 
                 'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean', 'spot_depth5_imbalance_mean', 'swap_depth5_imbalance_mean', 
                # 'spot_depth1_bid_ticks', 'spot_depth1_ask_ticks', 'swap_depth1_bid_ticks', 
                #  'swap_depth1_ask_ticks', 'spot_volatility_ticks', 'swap_volatility_ticks', 'spot_price_return_2min', 
                #  'swap_price_return_2min', 'spot_trade_volume_2min', 'spot_trade_count_2min', 'spot_bid_price_mean', 
                #  'spot_ask_price_mean', 'swap_bid_price_mean', 'swap_ask_price_mean', 'swap_trade_volume_2min', 
                #  'swap_trade_count_2min',  'basis_slippage_ticks', 
                #  'spot_midprice_mean', 'spot_midprice_std', 
                #  'spread_ticks', 'depth_imbalance_mean', 'volatility_ticks', 
                #  'trade_volume_2min', 'spot_midprice_open', 'spot_midprice_close', 'spot_midprice_high', 
                #  'spot_midprice_low', 'swap_midprice_mean', 'swap_midprice_std', 
                 'swap_buy_trade_ratio', 'spot_buy_trade_ratio',
                 'basis_mid_adjusted_mean_0', 'basis_mid_adjusted_std_0', 
                 'basis_mid_adjusted_mean_1', 'basis_mid_adjusted_std_1', 
                 'basis_mid_adjusted_mean_2', 'basis_mid_adjusted_std_2', 
                 'basis_mid_adjusted_mean_3', 'basis_mid_adjusted_std_3', 
                 'basis_mid_adjusted_mean_4', 'basis_mid_adjusted_std_4', 
                 'basis_mid_adjusted_mean_5', 'basis_mid_adjusted_std_5', 
                 'basis_mid_adjusted_mean_6', 'basis_mid_adjusted_std_6', 
                 'basis_mid_adjusted_mean_7', 'basis_mid_adjusted_std_7', 
                 'basis_mid_adjusted_mean_8', 'basis_mid_adjusted_std_8', 
                 'basis_mid_adjusted_mean_9', 'basis_mid_adjusted_std_9', 
                 'basis_mid_adjusted_mean_10', 'basis_mid_adjusted_std_10', 
                 'basis_mid_adjusted_mean_11', 'basis_mid_adjusted_std_11', 
                 'basis_mid_adjusted_mean_12', 
                 'basis_mid_adjusted_std_12', 
                 'basis_mid_adjusted_mean', 
                #  'basis_mid_adjusted_std'
                 ]
# Remove basis_mid_adjusted_std_{i} to avoid data leakage since they are calculated using future data
for i in range(13):
    selected_cols.remove(f'basis_mid_adjusted_std_{i}')

# selected_cols = ['gain_vs_threshold', 'basis_slippage', 
#                  'symbol', 'trade_mode', 'operation', 
#                  'exec_ts_utc', 'window_start_ms', 'window_end_ms', 
#                  'execute_delay_ms', 'timer_start_ts', 'taker_exec_ts', 
#                  'threshold', 'basis_expected', 'basis_executed', 
#                  'spot_midprice_mean', 'spot_midprice_std', 'spot_spread_mean', 
#                  'spot_midprice_open', 'spot_midprice_close', 'spot_midprice_high', 'spot_midprice_low', 
#                  'swap_midprice_mean', 'swap_midprice_std', 'swap_spread_mean', 
#                  'swap_spread_ticks', 'spot_spread_ticks', 
#                  'basis_ask_mean', 'basis_bid_mean', 'basis_ask_std', 'basis_bid_std', 
#                  'basis_ask_open', 'basis_bid_open', 'basis_ask_close', 'basis_bid_close', 
#                  'basis_ask_high', 'basis_bid_high', 'basis_ask_low', 'basis_bid_low', 
#                  'basis_ask_adjusted_mean', 'basis_bid_adjusted_mean', 'basis_ask_adjusted_std', 'basis_bid_adjusted_std', 
                #  'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean', 'spot_depth5_imbalance_mean', 'swap_depth5_imbalance_mean', 
#                  'spot_depth1_bid_ticks', 'spot_depth1_ask_ticks', 'swap_depth1_bid_ticks', 'swap_depth1_ask_ticks', 
#                  'spot_volatility_ticks', 'swap_volatility_ticks', 
#                  'spot_price_return_2min', 'swap_price_return_2min', 'spot_trade_volume_2min', 
#                  'spot_bid_price_mean', 'spot_ask_price_mean', 
#                  'spot_trade_count_2min', 'spot_buy_trade_ratio', 'swap_trade_volume_2min', 'swap_trade_count_2min', 
#                  'swap_bid_price_mean', 'swap_ask_price_mean', 'swap_buy_trade_ratio', 'basis_slippage_ticks'
#                  ]

# selected_cols = ['gain_vs_threshold', 'basis_slippage', 'operation',
# 'symbol', 'trade_mode', 'exec_ts_utc',
# 'execute_delay_ms', 'timer_start_ts', 'taker_exec_ts',
# 'threshold', 'basis_expected', 'basis_executed',
# 'spot_midprice_mean', 'spot_midprice_std', 'spot_spread_mean',
# 'spot_midprice_open', 'spot_midprice_close', 'spot_midprice_high',
# 'spot_midprice_low', 'swap_midprice_mean', 'swap_midprice_std',
# 'swap_spread_mean', 'swap_spread_ticks', 'spot_spread_ticks',
# 'basis_ask_mean', 'basis_bid_mean', 'basis_ask_open',
# 'basis_bid_open', 'basis_ask_close', 'basis_bid_close',
# 'basis_ask_high', 'basis_bid_high', 'basis_ask_low',
# 'basis_bid_low',
# 'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean',
# 'spot_depth1_bid_ticks', 'spot_depth1_ask_ticks', 'swap_depth1_bid_ticks',
# 'swap_depth1_ask_ticks', 'spot_volatility_ticks', 'swap_volatility_ticks',
# 'spot_price_return_2min', 'swap_price_return_2min', 'spot_trade_volume_2min',
# 'spot_trade_count_2min', 'spot_buy_trade_ratio', 'swap_trade_volume_2min',
# 'swap_trade_count_2min',
# ]

# Sort by exec_ts_utc
# combined_df['exec_ts_utc'] = pd.to_datetime(combined_df['exec_ts_utc'])
combined_df['exec_ts_utc'] = pd.to_datetime(
    combined_df['exec_ts_utc'],
    format='ISO8601',
    utc=True
)
combined_df = combined_df.sort_values('exec_ts_utc')

# print(combined_df['execute_delay_ms'].describe())

operation_select = 'open2'
# operation_select = 'close2'
delay_quantile = 90 # Filter out rows with execute_delay_ms above the 50th percentile (median) to focus on more typical cases. Adjust as needed (e.g., 80 for 80th percentile).
# Filter out outliers based on the 95th percentile of execute_delay_ms
upper_limit_delay = combined_df['execute_delay_ms'].quantile(delay_quantile/100)
print(f"{delay_quantile}th percentile of execute_delay_ms: {upper_limit_delay} ms")
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
# Select only the relevant columns and drop rows with missing values
# print(selected_cols)
print(filtered_df[selected_cols].describe())
df = filtered_df[selected_cols].dropna()
print(f"Final dataset shape after filtering and dropping NA: {df.shape}")
feature_cols = [
                # 'basis_mid_adjusted_mean', 
                # 'basis_mid_adjusted_std',
                #  'basis_mid_adjusted_mean_0', 'basis_mid_adjusted_std_0', 
#                  'basis_mid_adjusted_mean_1', 'basis_mid_adjusted_std_1', 
#                  'basis_mid_adjusted_mean_2', 'basis_mid_adjusted_std_2', 
#                  'basis_mid_adjusted_mean_3', 'basis_mid_adjusted_std_3', 
#                  'basis_mid_adjusted_mean_4', 'basis_mid_adjusted_std_4', 
#                  'basis_mid_adjusted_mean_5', 'basis_mid_adjusted_std_5', 
#                  'basis_mid_adjusted_mean_6', 'basis_mid_adjusted_std_6', 
#                  'basis_mid_adjusted_mean_7', 'basis_mid_adjusted_std_7', 
#                  'basis_mid_adjusted_mean_8', 'basis_mid_adjusted_std_8', 
#                  'basis_mid_adjusted_mean_9', 'basis_mid_adjusted_std_9', 
#                  'basis_mid_adjusted_mean_10', 'basis_mid_adjusted_std_10', 
#                  'basis_mid_adjusted_mean_11', 'basis_mid_adjusted_std_11', 
#                  'basis_mid_adjusted_mean_12', 'basis_mid_adjusted_std_12', 
                'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean', 'spot_depth5_imbalance_mean', 'swap_depth5_imbalance_mean', 
                'swap_buy_trade_ratio', 'spot_buy_trade_ratio',
                ]

weights = [-2, -1, -.5, -.1, -.05, -.01, 0, .01, .05, .1, .5, 1, 2]
# feature_cols = ['threshold', 
                # 'basis_expected', 
                #  'spot_midprice_mean', 'spot_spread_mean', 
                #  'spot_midprice_open', 'spot_midprice_close', 'spot_midprice_high', 'spot_midprice_low', 
                #  'swap_midprice_mean', 'swap_spread_mean', 
                #  'spot_midprice_std', 'swap_midprice_std', 
                #  'swap_spread_ticks', 'spot_spread_ticks', 
                #  'basis_ask_mean', 'basis_bid_mean', 
                #  'basis_ask_std', 'basis_bid_std', 
                #  'basis_ask_open', 'basis_bid_open', 'basis_ask_close', 'basis_bid_close', 
                #  'basis_ask_high', 'basis_bid_high', 'basis_ask_low', 'basis_bid_low', 
                #  'basis_ask_adjusted_mean', 'basis_bid_adjusted_mean', 
                #  'basis_ask_adjusted_std', 'basis_bid_adjusted_std', 
                #  'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean', 'spot_depth5_imbalance_mean', 'swap_depth5_imbalance_mean', 
                #  'spot_depth1_bid_ticks', 'spot_depth1_ask_ticks', 'swap_depth1_bid_ticks', 'swap_depth1_ask_ticks', 
                #  'spot_volatility_ticks', 'swap_volatility_ticks', 
                #  'spot_price_return_2min', 'swap_price_return_2min', 
                #  'spot_trade_volume_2min', 'swap_trade_volume_2min',
                #  'spot_bid_price_mean', 'spot_ask_price_mean', 
                #  'spot_trade_count_2min',  'swap_trade_count_2min', 
                #  'swap_bid_price_mean', 'swap_ask_price_mean', 
                #  'swap_buy_trade_ratio', 'spot_buy_trade_ratio', 
                #  ]


# Calculate columns
# df['basis_slippage_rate'] = (df['basis_executed'] - df['basis_expected']) / (df['basis_expected'] + 1e-12)
# df['basis_ask_k_volatility'] = (df['basis_ask_high'] - df['basis_ask_low']) / (np.abs(df['basis_ask_close'] - df['basis_ask_open']) + 1e-12)
# df['basis_bid_k_volatility'] = (df['basis_bid_high'] - df['basis_bid_low']) / (np.abs(df['basis_bid_close'] - df['basis_bid_open']) + 1e-12)

# basis_mid_adjusted_mean = (df['basis_ask_adjusted_mean'] + df['basis_bid_adjusted_mean']) / 2

basis_mid_mean = (df['basis_ask_mean'] + df['basis_bid_mean']) / 2
df['basis_mid_to_thres'] = (basis_mid_mean - df['threshold']) 
df['basis_mid_mean'] = basis_mid_mean
df['basis_adjusted_mid_to_thres'] = (df['basis_mid_adjusted_mean'] - df['threshold']) 
selected_feeture_cols = []
for i in range(13):
    df[f'basis_adj_mid_to_thres_{weights[i]}'] = (df[f'basis_mid_adjusted_mean_{i}'] - df['threshold']) 
    # feature_cols.append(f'basis_mid_adjusted_mean_{i}')
    feature_cols.append(f'basis_adj_mid_to_thres_{weights[i]}')
    selected_feeture_cols.append(f'basis_mid_adjusted_mean_{i}')

# print(filtered_df['exec_ts_utc'])
df['date'] = df['exec_ts_utc'].dt.date
print(df[selected_feeture_cols].describe())
# Compare daily values of IC and IR of basis_mid_adjusted_mean_{i} and basis_adj_mid_to_thres_{i}
result_IC_IR = []
import scipy.stats as stats
# Assume 'df' has 'factor_score' and 'future_return'
# Group by date to get daily cross-sectional correlation
def calculate_ic(group, factor_name, return_name):
    return stats.spearmanr(group[factor_name], group[return_name])[0]

ic_ir_list = [f'basis_mid_adjusted_mean_{i}' for i in range(13)]
ic_ir_list.append('basis_mid_mean')
result_IC_IR = []
for factor_name in ic_ir_list:
    label_name = 'gain_vs_threshold'
    # Calculate IC series
    daily_ic = df.groupby('date').apply(calculate_ic, factor_name=factor_name, return_name=label_name)

    # Calculate IR
    ic_mean = daily_ic.mean()
    ic_std = daily_ic.std()
    ir = ic_mean / ic_std
    print(f"IC of {factor_name}: {ic_mean:.8f} (+/- {ic_std:.8f}). IR: {ir:.8f}")
    result_IC_IR.append((ic_mean, ir))

# Plot IC and IR
plt.figure(figsize=(12, 10))
ic_values_mid = [x[0] for x in result_IC_IR]
ir_values_mid = [x[1] for x in result_IC_IR]
plt.subplot(2, 2, 1)
plt.plot(weights, ic_values_mid[:13], 'o-', label='IC: basis_mid_adjusted_mean')
# plt.plot(weights, ic_values_adj_mid, 's-', label='IC: basis_adj_mid_to_thres')
plt.xlabel('Index')
plt.ylabel('IC')
plt.title('IC Values of Adjusted Basis Mids')
plt.grid(True)
plt.legend()


plt.subplot(2, 2, 2)
plt.plot(weights, ir_values_mid[:13], 'o-', label='IR: basis_mid_adjusted_mean')
# plt.plot(weights, ir_values_adj_mid, 's-', label='IR: basis_adj_mid_to_thres')
plt.xlabel('Index')
plt.ylabel('IR')
plt.title('IR Values of Adjusted Basis Mids')
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

df['const.'] = 1.0
feature_cols.append('const.')

# df['basis_executed_to_thres'] = (df['basis_executed'] - df['threshold'])
# feature_cols.append('basis_executed_to_thres')

print(f"Selected {len(feature_cols)} features")

# Visualize correlation
label_names = ['gain_vs_threshold']
df_corr = df[feature_cols + label_names]#.corr()
corr_matrix = df_corr.corr()


# 3. Plot using Seaborn
# plt.figure(figsize=(16, 12))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()

# Plot mean of f"basis_mid_adjusted_mean_{i}" for i in range(13) against i
plt.subplot(2, 2, 3)
for i in range(13):
    plt.plot(weights[i], df[f'basis_mid_adjusted_mean_{i}'].mean(), 'bo')
plt.xlabel('Index')
plt.ylabel('Mean of basis_mid_adjusted_mean')
plt.title('Mean Values of Adjusted Basis Mids')
plt.grid(True)
plt.subplot(2, 2, 4)
for i in range(13):
    plt.plot(weights[i], df[f'basis_mid_adjusted_mean_{i}'].std(), 'bo')
plt.xlabel('Index')
plt.ylabel('Standard Deviation of basis_mid_adjusted_mean')
plt.title('Standard Deviation Values of Adjusted Basis Mids')
plt.grid(True)
plt.savefig(f'output/ic_ir_{symbol}_{label_name}_nMod{len(models)}_{label_name}_delay{delay_precentile}_{operation}_nml{normalize_X}_mode{mode}.png', bbox_inches='tight')

plt.show()


# Prepare data for modeling
df_sample = df.copy()
X = df_sample[feature_cols].values
print(f"Feature matrix shape: {X.shape}")
print(df_sample[feature_cols].head())

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

if 'OLS Regression' in models:
    results['OLS Regression'] = {}
    # Train an OLS regression model using statsmodels
    # X_train_sm = sm.add_constant(X_train)  # Add intercept term
    X_train_sm = X_train.copy()  # Ensure it's a pure numpy array
    model_ols = sm.OLS(y_train, X_train_sm).fit()
    # X_test_sm = sm.add_constant(X_test)
    X_test_sm = X_test.copy()  # Ensure it's a pure numpy array
    y_pred_ols_norm = model_ols.predict(X_test_sm)
    y_pred_ols = y_pred_ols_norm * y_train_std + y_train_mean# Denormalize predictions
    y_pred_ols_train_norm = model_ols.predict(X_train_sm)
    y_pred_ols_train = y_pred_ols_train_norm * y_train_std + y_train_mean  # Denormalize train predictions
    label_pred = y_pred_ols - beta * x_thres  

    print(model_ols.summary())
    # importance_ols = model_ols.params[0:]  # Exclude intercept
    importance_ols = model_ols.tvalues[0:]
    indices_ols = np.argsort(np.abs(importance_ols))[::-1]
    mse_ols = mean_squared_error(y_test, y_pred_ols)
    r2_ols = r2_score(y_test, y_pred_ols)
    print(f"OLS Regression - MSE: {mse_ols:.4f}, R2: {r2_ols:.4f}")
    results['OLS Regression']['MSE'] = mse_ols
    results['OLS Regression']['R2'] = r2_ols
    results['OLS Regression']['y_pred'] = y_pred_ols
    results['OLS Regression']['y_pred_train'] = y_pred_ols_train
    results['OLS Regression']['importance'] = importance_ols
    results['OLS Regression']['indices'] = indices_ols
    results['OLS Regression']['label_pred'] = label_pred


if 'Linear Regression' in models:
    results['Linear Regression'] = {}
    # Train a linear regression model
    model_LR = LinearRegression()
    model_LR.fit(X_train, y_train)
    y_pred_LR = model_LR.predict(X_test)
    y_pred_LR = y_pred_LR * y_train_std + y_train_mean# Denormalize predictions
    y_pred_LR_train = model_LR.predict(X_train) * y_train_std + y_train_mean  # Denormalize train predictions
    label_pred_LR = y_pred_LR - beta * x_thres
    mse_LR = mean_squared_error(y_test, y_pred_LR)
    r2_LR = r2_score(y_test, y_pred_LR)
    print(f"Linear Regression - MSE: {mse_LR:.4f}, R2: {r2_LR:.4f}")

    # Feature importance for linear regression (absolute value of coefficients)
    print(model_LR.coef_.shape)
    coef_importance = model_LR.coef_
    indices_lr = np.argsort(np.abs(coef_importance))[::-1]
    results['Linear Regression']['MSE'] = mse_LR
    results['Linear Regression']['R2'] = r2_LR
    results['Linear Regression']['y_pred'] = y_pred_LR
    results['Linear Regression']['y_pred_train'] = y_pred_LR_train
    results['Linear Regression']['importance'] = coef_importance
    results['Linear Regression']['indices'] = indices_lr
    results['Linear Regression']['label_pred'] = label_pred_LR


if 'LightGBM Regressor' in models:
    # Train a lightGBM model
    model_lgb = lgb.LGBMRegressor(n_estimators=10000, reg_alpha=0.5, 
                                  max_depth=20, random_state=42,
                                  verbosity = -1)
    model_lgb.fit(X_train, y_train)
    y_pred_lgb = model_lgb.predict(X_test)
    y_pred_lgb = y_pred_lgb * y_train_std + y_train_mean  # Denormalize predictions
    y_pred_lgb_train = model_lgb.predict(X_train) * y_train_std + y_train_mean  # Denormalize train predictions
    label_pred_lgb = y_pred_lgb - beta * x_thres
    mse_lgb = mean_squared_error(y_test, y_pred_lgb)
    r2_lgb = r2_score(y_test, y_pred_lgb)
    lgb_importances = model_lgb.feature_importances_
    indices_lgb = np.argsort(lgb_importances)[::-1]
    print(f"LightGBM Regressor - MSE: {mse_lgb:.4f}, R2: {r2_lgb:.4f}")
    results['LightGBM Regressor'] = {}
    results['LightGBM Regressor']['MSE'] = mse_lgb
    results['LightGBM Regressor']['R2'] = r2_lgb
    results['LightGBM Regressor']['y_pred'] = y_pred_lgb
    results['LightGBM Regressor']['y_pred_train'] = y_pred_lgb_train
    results['LightGBM Regressor']['importance'] = lgb_importances
    results['LightGBM Regressor']['indices'] = indices_lgb
    results['LightGBM Regressor']['label_pred'] = label_pred_lgb


if 'XGBoost Regressor' in models:
    results['XGBoost Regressor'] = {}
    # Train a XGBoost model
    model_xgb = xgb.XGBRegressor(n_estimators=20000, max_depth=10,
                                 device='cuda',
                                 random_state=42)
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_test)
    y_pred_xgb = y_pred_xgb * y_train_std + y_train_mean  # Denormalize predictions
    y_pred_xgb_train = model_xgb.predict(X_train) * y_train_std + y_train_mean  # Denormalize train predictions
    label_pred_xgb = y_pred_xgb - beta * x_thres
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)
    importance_xgb = model_xgb.feature_importances_
    indices_xgb = np.argsort(importance_xgb)[::-1]
    print(f"XGBoost Regressor - MSE: {mse_xgb:.4f}, R2: {r2_xgb:.4f}")
    results['XGBoost Regressor']['MSE'] = mse_xgb
    results['XGBoost Regressor']['R2'] = r2_xgb
    results['XGBoost Regressor']['y_pred'] = y_pred_xgb
    results['XGBoost Regressor']['y_pred_train'] = y_pred_xgb_train
    results['XGBoost Regressor']['importance'] = importance_xgb
    results['XGBoost Regressor']['indices'] = indices_xgb
    results['XGBoost Regressor']['label_pred'] = label_pred_xgb

if 'Random Forest Regressor' in models:
    results['Random Forest Regressor'] = {}
    # Train a Random Forest Regressor
    model_rf = RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    y_pred_rf = y_pred_rf * y_train_std + y_train_mean  # Denormalize predictions
    y_pred_rf_train = model_rf.predict(X_train) * y_train_std + y_train_mean  # Denormalize train predictions
    label_pred_rf = y_pred_rf - beta * x_thres
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"Random Forest Regressor - MSE: {mse_rf:.4f}, R2: {r2_rf:.4f}")

    # Feature importance plot
    importances = model_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    results['Random Forest Regressor']['MSE'] = mse_rf
    results['Random Forest Regressor']['R2'] = r2_rf
    results['Random Forest Regressor']['y_pred'] = y_pred_rf
    results['Random Forest Regressor']['y_pred_train'] = y_pred_rf_train
    results['Random Forest Regressor']['importance'] = importances
    results['Random Forest Regressor']['indices'] = indices
    results['Random Forest Regressor']['label_pred'] = label_pred_rf


# Plot feature importance for all models
models_plot_imp = [model for model in models if 'importance' in results[model]]  # Exclude CNN from importance plot
nplot_1 = len(models_plot_imp)
fig, axes = plt.subplots(1, nplot_1, figsize=(24, 6))
for i in range(nplot_1):
    axes[i].grid(True)
    axes[i].set_title(f"{models_plot_imp[i]} (sym: {symbol}, {label_name}),\n R2: {results[models_plot_imp[i]]['R2']:.4f}")
    # print(f"X_train shape: {X_train.shape}, importance shape: {results[models_plot_imp[i]]['importance'].shape}")
    axes[i].bar(range(X_train.shape[1]), results[models_plot_imp[i]]['importance'][results[models_plot_imp[i]]['indices']], align="center")
    axes[i].set_xticks(range(X_train.shape[1]))
    axes[i].set_xticklabels([feature_cols[j] for j in results[models_plot_imp[i]]['indices']], rotation=90)
    axes[i].set_xlim([-1, X_train.shape[1]])
    axes[i].set_ylabel("Importance")
    axes[i].set_xlabel("Feature")

plt.tight_layout()
plt.savefig(f'output/ft_imp_{symbol}_{label_name}_nMod{len(models)}_{label_name}_nFts{X_train.shape[1]}{delay_exec}_delay{delay_precentile}_{operation}_nml{normalize_X}_mode{mode}.png', bbox_inches='tight')

plt.show()

# fig2, axes2 = plt.subplots(len(models), 2, figsize=(18, 16))

# # Denormalize y_train for plotting
# y_train_denorm = y_train * y_train_std + y_train_mean

# for i, model_name in enumerate(models):
#     y_pred = results[model_name]['y_pred']
#     y_pred_train = results[model_name]['y_pred_train']
#     axes2[i,0].scatter(y_test, y_pred, alpha=0.5, label=model_name)
#     axes2[i,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
#     axes2[i,0].set_xlabel('True Values')
#     axes2[i,0].set_ylabel('Predicted Values')
#     axes2[i,0].set_title(f'{model_name}: True vs Pred (sym: {symbol}, {label_name})')
#     axes2[i,0].grid(True)
#     axes2[i,0].legend()
#     axes2[i,1].scatter(y_train_denorm, y_pred_train, alpha=0.5, label=f'{model_name} (Train)', color='orange')
#     axes2[i,1].plot([y_train_denorm.min(), y_train_denorm.max()], [y_train_denorm.min(), y_train_denorm.max()], 'k--', lw=2)
#     axes2[i,1].set_xlabel('True Values (Train)')
#     axes2[i,1].set_ylabel('Predicted Values')
#     axes2[i,1].set_title(f'{model_name}: True vs Pred (Train) (sym: {symbol}, {label_name})')
#     axes2[i,1].grid(True)
#     axes2[i,1].legend()

# plt.tight_layout()
# plt.savefig(f'output/pred_{symbol}_{label_name}_nMod{len(models)}_nFts{X_train.shape[1]}{delay_exec}_delay{delay_precentile}_{operation}_nml{normalize_X}_mode{mode}.png', bbox_inches='tight')

# plt.show()

