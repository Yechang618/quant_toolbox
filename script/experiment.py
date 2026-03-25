####
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

# Define the folder path
mode = 2
folder_path = f'dataset/preprocessed/mode{mode}/'
# folder_path = 'dataset/preprocessed/mode2/'

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Read and combine all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

print(f"Combined {len(csv_files)} files")
print(f"Total rows: {len(combined_df)}")

print(combined_df.info())

selected_cols = ['gain_vs_threshold', 'basis_slippage', 
                # 'symbol', 'trade_mode', 'exec_ts_utc', 
                'execute_delay_ms', 'timer_start_ts', 'taker_exec_ts', 
                'threshold', 'basis_expected', 'basis_executed', 
                'spot_midprice_mean', 'spot_midprice_std', 'spot_spread_mean', 
                'spot_midprice_open', 'spot_midprice_close', 'spot_midprice_high', 
                'spot_midprice_low', 'swap_midprice_mean', 'swap_midprice_std', 
                'swap_spread_mean', 'swap_spread_ticks', 'spot_spread_ticks', 
                'basis_ask_mean', 'basis_bid_mean', 'basis_ask_open', 
                'basis_bid_open', 'basis_ask_close', 'basis_bid_close', 
                'basis_ask_high', 'basis_bid_high', 'basis_ask_low', 
                'basis_bid_low', 
                'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean', 
                'spot_depth1_bid_ticks', 'spot_depth1_ask_ticks', 'swap_depth1_bid_ticks', 
                'swap_depth1_ask_ticks', 'spot_volatility_ticks', 'swap_volatility_ticks', 
                'spot_price_return_60s', 'swap_price_return_60s', 'spot_trade_volume_60s', 
                'spot_trade_count_60s', 'spot_buy_trade_ratio', 'swap_trade_volume_60s', 
                'swap_trade_count_60s', 
                # 'swap_buy_trade_ratio', 
                'spot_basis_slippage_ticks', 
                # 'spread_ticks', 'depth_imbalance_mean', 'volatility_ticks', 'trade_volume_60s'
                ]
# Sort by exec_ts_utc
# combined_df['exec_ts_utc'] = pd.to_datetime(combined_df['exec_ts_utc'])
combined_df['exec_ts_utc'] = pd.to_datetime(
    combined_df['exec_ts_utc'],
    format='ISO8601',
    utc=True
)
combined_df = combined_df.sort_values('exec_ts_utc')
print(combined_df['execute_delay_ms'].describe())

# Filter out outliers based on the 95th percentile of execute_delay_ms
upper_limit_delay = combined_df['execute_delay_ms'].quantile(0.50)
print(f"50th percentile of execute_delay_ms: {upper_limit_delay} ms")
filtered_df = combined_df[combined_df['execute_delay_ms'] <= upper_limit_delay]

# Filter out outliers based on the 10th and 90th percentiles of gain_vs_threshold
lower_limit_gain = filtered_df['gain_vs_threshold'].quantile(0.30)
upper_limit_gain = filtered_df['gain_vs_threshold'].quantile(0.70)
print(f"30th percentile of gain_vs_threshold: {lower_limit_gain}")
print(f"70th percentile of gain_vs_threshold: {upper_limit_gain}")
filtered_df = filtered_df[(filtered_df['gain_vs_threshold'] >= lower_limit_gain) & (filtered_df['gain_vs_threshold'] <= upper_limit_gain)]

# Select only the relevant columns and drop rows with missing values
df = filtered_df[selected_cols].dropna()
feature_cols = ['threshold', 'basis_expected', 
                'spot_midprice_mean', 'spot_midprice_std', 'spot_spread_mean', 
                'spot_midprice_open', #'spot_midprice_close', 'spot_midprice_high', 
                # 'spot_midprice_low', 'swap_midprice_mean', 'swap_midprice_std', 
                # 'swap_spread_mean', 'swap_spread_ticks', 'spot_spread_ticks', 
                # 'basis_ask_mean', 'basis_bid_mean', 'basis_ask_open', 
                # 'basis_bid_open', 'basis_ask_close', 'basis_bid_close', 
                # 'basis_ask_high', 'basis_bid_high', 'basis_ask_low', 
                # 'basis_bid_low', 
                'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean', 
                'spot_depth1_bid_ticks', 'spot_depth1_ask_ticks', 'swap_depth1_bid_ticks', 
                'swap_depth1_ask_ticks', 'spot_volatility_ticks', 'swap_volatility_ticks', 
                'spot_price_return_60s', 'swap_price_return_60s', 'spot_trade_volume_60s', 
                'spot_trade_count_60s', 'spot_buy_trade_ratio', 'swap_trade_volume_60s', 
                'swap_trade_count_60s', 
                # 'spot_basis_slippage_ticks', 
                ]
label_cols = ['gain_vs_threshold']#, 'basis_slippage']
print(f"Selected {len(feature_cols)} features and {len(label_cols)} labels")

## Calculate columns
# df['basis_slippage_rate'] = (df['basis_executed'] - df['basis_expected']) / (df['basis_expected'] + 1e-12)
df['basis_ask_k_volatility'] = (df['basis_ask_high'] - df['basis_ask_low']) / (np.abs(df['basis_ask_close'] - df['basis_ask_open']) + 1e-12)
df['basis_bid_k_volatility'] = (df['basis_bid_high'] - df['basis_bid_low']) / (np.abs(df['basis_bid_close'] - df['basis_bid_open']) + 1e-12)
feature_cols.append('basis_ask_k_volatility')
feature_cols.append('basis_bid_k_volatility')
## Prepare data for modeling
df_sample = df.copy()
X = df_sample[feature_cols].values

# Generate labels based on the selected label column
label_name = 'gain_vs_threshold'
# label_name = 'basis_slippage'
# label_name = 'basis_slippage_rate'
y = df_sample[label_name].values  


y = np.squeeze(y)  # Convert to 1D array if it's a single column
# from sklearn.model_selection import train_test_split
test_size = 0.2
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test = X[:int(len(X) * (1 - test_size))], X[int(len(X) * (1 - test_size)):]
y_train, y_test = y[:int(len(y) * (1 - test_size))], y[int(len(y) * (1 - test_size)):]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")  
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Normalize features and labels
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
y_train_mean = y_train.mean()
y_train_std = y_train.std()
X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)
y_train = (y_train - y_train_mean) / (y_train_std + 1e-8)

X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)

# Train a lightGBM model
# import lightgbm as lgb
# model_lgb = lgb.LGBMRegressor(n_estimators=100, random_state=42)
# model_lgb.fit(X_train, y_train)
# y_pred_lgb = model_lgb.predict(X_test)
# y_pred_lgb = y_pred_lgb * y_train_std + y_train_mean  # Denormalize predictions
# mse_lgb = mean_squared_error(y_test, y_pred_lgb)
# r2_lgb = r2_score(y_test, y_pred_lgb)
# print(f"LightGBM Regressor - MSE: {mse_lgb:.4f}, R2: {r2_lgb:.4f}")

# Train a Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
model_rf = RandomForestRegressor(n_estimators=1000, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
y_pred_rf = y_pred_rf * y_train_std + y_train_mean  # Denormalize predictions
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest Regressor - MSE: {mse_rf:.4f}, R2: {r2_rf:.4f}")
# Feature importance plot
importances = model_rf.feature_importances_
indices = np.argsort(importances)[::-1]


# Train a linear regression model
from sklearn.linear_model import LinearRegression
model_LR = LinearRegression()
model_LR.fit(X_train, y_train)
y_pred_LR = model_LR.predict(X_test)
y_pred_LR = y_pred_LR * y_train_std + y_train_mean  # Denormalize predictions
mse_LR = mean_squared_error(y_test, y_pred_LR)
r2_LR = r2_score(y_test, y_pred_LR)
print(f"Linear Regression - MSE: {mse_LR:.4f}, R2: {r2_LR:.4f}")

# Feature importance for linear regression (absolute value of coefficients)
print(model_LR.coef_.shape)
# coef_importance = np.abs(model_LR.coef_)
coef_importance = model_LR.coef_
indices_lr = np.argsort(coef_importance)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
# Plot feature importance for Random Forest
axes[0].set_title("Random Forest Feature Importances")
axes[0].bar(range(X_train.shape[1]), importances[indices], align="center")
axes[0].set_xticks(range(X_train.shape[1]))
axes[0].set_xticklabels([feature_cols[i] for i in indices], rotation=90)
axes[0].set_xlim([-1, X_train.shape[1]])
axes[0].set_ylabel("Importance")
axes[0].set_xlabel("Feature")
# Plot feature importance for Linear Regression
axes[1].set_title("Linear Regression Coefficient Importances")
axes[1].bar(range(X_train.shape[1]), coef_importance[indices_lr], align="center")
axes[1].set_xticks(range(X_train.shape[1]))
axes[1].set_xticklabels([feature_cols[i] for i in indices_lr], rotation=90)
axes[1].set_xlim([-1, X_train.shape[1]])
axes[1].set_ylabel("Coefficient Value")
axes[1].set_xlabel("Feature")
# Plot feature importance for LightGBM
# lgb_importances = model_lgb.feature_importances_
# indices_lgb = np.argsort(lgb_importances)[::-1]
# axes[2].set_title("LightGBM Feature Importances")
# axes[2].bar(range(X_train.shape[1]), lgb_importances[indices_lgb], align="center")
# axes[2].set_xticks(range(X_train.shape[1]))
# axes[2].set_xticklabels([feature_cols[i] for i in indices_lgb], rotation=90)
# axes[2].set_xlim([-1, X_train.shape[1]])   
plt.tight_layout()

plt.savefig(f'feature_importance_label_{label_name}_mode{mode}.png', bbox_inches='tight')
plt.show()  
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))

# Plot true vs predicted values for Random Forest
axes2[0].scatter(y_test, y_pred_rf, alpha=0.5, label='Random Forest')
axes2[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes2[0].set_xlabel('True Values')
axes2[0].set_ylabel('Predicted Values')
axes2[0].set_title('Random Forest: True vs Predicted Values')
axes2[0].grid(True)
axes2[0].legend()

# Plot true vs predicted values for Linear Regression
axes2[1].scatter(y_test, y_pred_LR, alpha=0.5, label='Linear Regression')
axes2[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes2[1].set_xlabel('True Values')
axes2[1].set_ylabel('Predicted Values')
axes2[1].set_title('Linear Regression: True vs Predicted Values')
axes2[1].grid(True)
axes2[1].legend()

plt.tight_layout()
plt.savefig(f'true_vs_predicted_label_{label_name}_mode{mode}.png', bbox_inches='tight')
plt.show()
