import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import seaborn as sns

results = {}
models = ['OLS Regression', 'Linear Regression',  
          'LightGBM Regressor', 'XGBoost Regressor', 
          'Random Forest Regressor', 'CNN Regressor', 
          ]

TOLERENCE = 1e-12

mode = 2
delay_exec = ''
# delay_exec = '_2'
normalize_X = 0
operation = 'open2'
# operation = 'close2'
delay_precentile = 30
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

selected_cols = ['gain_vs_threshold', 'basis_slippage', 'operation',
'symbol', 'trade_mode', 'exec_ts_utc',
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
# 'spot_basis_slippage_ticks',
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

# print(combined_df['execute_delay_ms'].describe())

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
# Select only the relevant columns and drop rows with missing values
df = filtered_df[selected_cols].dropna()

feature_cols = ['threshold', 
                'basis_expected', 
                # 'spot_midprice_mean', 
                'spot_midprice_std', 
                # 'spot_spread_mean', 
                'spot_midprice_open', 'spot_midprice_close', 'spot_midprice_high', 
                 'spot_midprice_low', 'swap_midprice_mean', 'swap_midprice_std', 
                 'swap_spread_mean', 'swap_spread_ticks', 'spot_spread_ticks', 
                 'basis_ask_mean', 'basis_bid_mean', 'basis_ask_open', 
                 'basis_bid_open', 
                'basis_ask_close', 'basis_bid_close', 
                 'basis_ask_high', 'basis_bid_high', 'basis_ask_low', 
                 'basis_bid_low', 
                'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean', 
                # 'spot_depth1_bid_ticks', 'spot_depth1_ask_ticks', 'swap_depth1_bid_ticks', 
                # 'swap_depth1_ask_ticks', 'spot_volatility_ticks', 'swap_volatility_ticks', 
                'spot_price_return_60s', 'swap_price_return_60s', 
                'spot_trade_volume_60s', 
                'spot_trade_count_60s', 
                'spot_buy_trade_ratio', 
                'swap_trade_volume_60s', 
                'swap_trade_count_60s', 
                # 'basis_slippage_rate'
                # 'spot_basis_slippage_ticks', 
                ]
print(f"Selected {len(feature_cols)} features")

# Calculate columns
df['basis_slippage_rate'] = (df['basis_executed'] - df['basis_expected']) / (df['basis_expected'] + 1e-12)
df['basis_ask_k_volatility'] = (df['basis_ask_high'] - df['basis_ask_low']) / (np.abs(df['basis_ask_close'] - df['basis_ask_open']) + 1e-12)
df['basis_bid_k_volatility'] = (df['basis_bid_high'] - df['basis_bid_low']) / (np.abs(df['basis_bid_close'] - df['basis_bid_open']) + 1e-12)

basis_mid_mean = (df['basis_ask_mean'] + df['basis_bid_mean']) / 2
df['basis_mid_to_thres'] = (basis_mid_mean - df['threshold']) 
basis_close_mid = (df['basis_ask_close'] + df['basis_bid_close']) / 2
df['basis_close_to_thres'] = (basis_close_mid - df['threshold'])
df['basis_expected_to_thres'] = (df['basis_expected'] - df['threshold'])
df['basis_ask_close_to_thres'] = (df['basis_ask_close'] - df['threshold'])
df['basis_bid_close_to_thres'] = (df['basis_bid_close'] - df['threshold'])

# feature_cols.append('basis_expected_to_thres')
# feature_cols.append('basis_ask_k_volatility')
# feature_cols.append('basis_bid_k_volatility')
# feature_cols.append('basis_close_to_thres')
# feature_cols.append('basis_mid_to_thres')
# feature_cols.append('basis_ask_close_to_thres')
# feature_cols.append('basis_bid_close_to_thres')

df['const.'] = 1.0
feature_cols.append('const.')

# df['basis_executed_to_thres'] = (df['basis_executed'] - df['threshold'])
# feature_cols.append('basis_executed_to_thres')

# Visualize correlation
label_names = ['basis_slippage', 'gain_vs_threshold']
df_corr = df[feature_cols + label_names].corr()
corr_matrix = df_corr.corr()
# 3. Plot using Seaborn
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Prepare data for modeling
df_sample = df.copy()
X = df_sample[feature_cols].values

# Generate labels based on the selected label column

Y = df_sample[[label_name, 'gain_vs_threshold']].values
# y = np.squeeze(y)  # Convert to 1D array if it's a single column

# Fix: Remove redundant train_test_split and use consistent test_size
test_size = 0.2  # Changed from 0.1 to 0.2 for better evaluation
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Removed redundant line
X_train, X_test = X[:int(len(X) * (1 - test_size))], X[int(len(X) * (1 - test_size)):]
Y_train, Y_test = Y[:int(len(Y) * (1 - test_size))], Y[int(len(Y) * (1 - test_size)):]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")

x_thres = X_test[:, feature_cols.index('threshold')]
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
    importance_ols = model_ols.params[0:]  # Exclude intercept
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

if 'CNN Regressor' in models:
    results['CNN Regressor'] = {}
    # ==================== Train a DEEPER CNN model ====================
    # 更深的 CNN 结构 - 输入维度 (n_sample, n_feature)
    # 内部转换为 (n_sample, n_feature, 1) 用于 Conv1D
    n_features = X_train.shape[1]

    model_cnn = keras.Sequential([
        # Input layer - 输入形状 (n_features, 1)
        layers.Input(shape=(n_features, 1)),
        
        # 第一组卷积块
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 第二组卷积块
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 第三组卷积块
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 第四组卷积块
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 全局池化
        layers.GlobalAveragePooling1D(),
        
        # 全连接层
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        # 输出层
        layers.Dense(1)
    ])

    model_cnn.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model_cnn.summary()

    # 训练模型 - 输入需要扩展为 3D (n_sample, n_features, 1)
    history = model_cnn.fit(
        X_train[..., np.newaxis], 
        y_train, 
        epochs=50, 
        batch_size=256, 
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
    )

    # 预测
    y_pred_cnn_norm = model_cnn.predict(X_test[..., np.newaxis]).flatten()
    y_pred_cnn = y_pred_cnn_norm * y_train_std + y_train_mean  # Denormalize predictions

    y_pred_cnn_train_norm = model_cnn.predict(X_train[..., np.newaxis]).flatten()
    y_pred_cnn_train = y_pred_cnn_train_norm * y_train_std + y_train_mean  # Denormalize train predictions
    label_pred_cnn = y_pred_cnn - beta * x_thres

    mse_cnn = mean_squared_error(y_test, y_pred_cnn)
    r2_cnn = r2_score(y_test, y_pred_cnn)
    print(f"CNN - MSE: {mse_cnn:.4f}, R2: {r2_cnn:.4f}")
    results['CNN Regressor']['MSE'] = mse_cnn
    results['CNN Regressor']['R2'] = r2_cnn
    results['CNN Regressor']['y_pred'] = y_pred_cnn
    results['CNN Regressor']['y_pred_train'] = y_pred_cnn_train
    results['CNN Regressor']['label_pred'] = label_pred_cnn

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
    axes[i].set_title(f"{models_plot_imp[i]} Feature Imp. (sym: {symbol}, {label_name}),\n MSE: {results[models_plot_imp[i]]['MSE']:.4f}, R2: {results[models_plot_imp[i]]['R2']:.4f}")
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

fig2, axes2 = plt.subplots(len(models), 2, figsize=(18, 16))

# Denormalize y_train for plotting
y_train_denorm = y_train * y_train_std + y_train_mean

for i, model_name in enumerate(models):
    y_pred = results[model_name]['y_pred']
    y_pred_train = results[model_name]['y_pred_train']
    axes2[i,0].scatter(y_test, y_pred, alpha=0.5, label=model_name)
    axes2[i,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    axes2[i,0].set_xlabel('True Values')
    axes2[i,0].set_ylabel('Predicted Values')
    axes2[i,0].set_title(f'{model_name}: True vs Pred (sym: {symbol}, {label_name})')
    axes2[i,0].grid(True)
    axes2[i,0].legend()
    axes2[i,1].scatter(y_train_denorm, y_pred_train, alpha=0.5, label=f'{model_name} (Train)', color='orange')
    axes2[i,1].plot([y_train_denorm.min(), y_train_denorm.max()], [y_train_denorm.min(), y_train_denorm.max()], 'k--', lw=2)
    axes2[i,1].set_xlabel('True Values (Train)')
    axes2[i,1].set_ylabel('Predicted Values')
    axes2[i,1].set_title(f'{model_name}: True vs Pred (Train) (sym: {symbol}, {label_name})')
    axes2[i,1].grid(True)
    axes2[i,1].legend()

plt.tight_layout()
plt.savefig(f'output/pred_{symbol}_{label_name}_nMod{len(models)}_nFts{X_train.shape[1]}{delay_exec}_delay{delay_precentile}_{operation}_nml{normalize_X}_mode{mode}.png', bbox_inches='tight')
plt.show()

fig3, axes3 = plt.subplots(1, len(models), figsize=(18, 6))
for i, model_name in enumerate(models):
    label_pred = results[model_name]['label_pred']
    axes3[i].scatter(label_test, label_pred, alpha=0.5, label=model_name)
    axes3[i].set_xlabel('Threshold')
    axes3[i].set_ylabel('Predicted Label (Pred - beta * Threshold)')
    axes3[i].set_title(f'{model_name}: Predicted Label vs Threshold ({label_name})')
    axes3[i].grid(True)
    axes3[i].legend()
plt.tight_layout()
plt.savefig(f'output/label_pred_{symbol}_{label_name}_nMod{len(models)}_nFts{X_train.shape[1]}{delay_exec}_delay{delay_precentile}_{operation}_nml{normalize_X}_mode{mode}.png', bbox_inches='tight')
plt.show()