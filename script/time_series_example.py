import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from math import sqrt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 设置 PyTorch 单线程模式
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from prophet import Prophet

np.random.seed(42)
torch.manual_seed(42)

bright_colors = ["#D81B60", "#1E88E5", "#FFC107", "#43A047", "#8E24AA", "#F4511E", "#00ACC1", "#7CB342", "#FB8C00", "#3949AB"]

# 1) 生成虚拟时间序列数据
def generate_synthetic_data(n=1000, start_date="2018-01-01"):
    dates = pd.date_range(start=start_date, periods=n, freq="D")
    t = np.arange(n)

    # 趋势项(缓慢上升)
    trend = 0.03 * t

    # 季节项:周季节(周期=7)，年季节(周期~365)
    weekly = 5 * np.sin(2 * np.pi * t / 7)
    yearly = 3 * np.sin(2 * np.pi * t / 365) + 2 * np.cos(2 * np.pi * t / 365)

    # 外生变量:营销活动(稀疏脉冲)
    marketing = np.zeros(n)
    spike_days = np.random.choice(np.arange(30, n-30), size=12, replace=False)
    for d in spike_days:
        marketing[d:d+3] += np.random.uniform(5, 15)

    # 节假日冲击(随机选择一些日期)
    holiday = np.zeros(n)
    holiday_days = np.random.choice(np.arange(20, n-20), size=10, replace=False)
    for d in holiday_days:
        holiday[d] += np.random.uniform(8, 20)

    # 噪声与离群点
    noise = np.random.normal(0, 2.0, size=n)
    outliers_idx = np.random.choice(np.arange(50, n-50), size=8, replace=False)
    noise[outliers_idx] += np.random.uniform(10, 20, size=len(outliers_idx)) * np.random.choice([1, -1], size=len(outliers_idx))

    # 真值构造
    y = 40 + trend + weekly + yearly + 0.5 * marketing + 0.8 * holiday + noise

    df = pd.DataFrame({
        "ds": dates,
        "y": y,
        "trend": trend,
        "weekly": weekly,
        "yearly": yearly,
        "marketing": marketing,
        "holiday": holiday
    })

    # 常用时间特征
    df["dow"] = df["ds"].dt.dayofweek
    df["sin_w"] = np.sin(2*np.pi*df["dow"]/7)
    df["cos_w"] = np.cos(2*np.pi*df["dow"]/7)
    dayofyear = df["ds"].dt.dayofyear
    df["sin_y"] = np.sin(2*np.pi*dayofyear/365.25)
    df["cos_y"] = np.cos(2*np.pi*dayofyear/365.25)
    return df

df = generate_synthetic_data(n=1000)

# 切分:训练/验证/测试
n = len(df)
train_end = int(n * 0.7)      # 0~699
val_end = int(n * 0.85)       # 700~849
test_end = n                  # 850~999

df_train = df.iloc[:train_end].copy()
df_val   = df.iloc[train_end:val_end].copy()
df_test  = df.iloc[val_end:test_end].copy()

H = 30  # 多步预测长度
W = 60  # LSTM输入窗口

# 2) 可视化图1:序列与分解成分
plt.figure(figsize=(12, 6))
plt.plot(df["ds"], df["y"], color=bright_colors[1], lw=1.5, label="实际序列 y")
plt.plot(df["ds"], df["trend"]+40, color=bright_colors[0], lw=1.5, alpha=0.9, label="趋势(移位以便观察)")
plt.plot(df["ds"], df["weekly"], color=bright_colors[2], lw=1.2, alpha=0.9, label="周季节")
plt.plot(df["ds"], df["yearly"], color=bright_colors[3], lw=1.2, alpha=0.9, label="年季节")
plt.scatter(df["ds"].iloc[::50], df["y"].iloc[::50], color=bright_colors[6], s=12, alpha=0.7, label="稀疏采样点")
plt.axvspan(df_val["ds"].iloc[0], df_val["ds"].iloc[-1], color="#80DEEA", alpha=0.2, label="验证区间")
plt.axvspan(df_test["ds"].iloc[0], df_test["ds"].iloc[-1], color="#FFCC80", alpha=0.2, label="测试区间")
plt.title("图1:时间序列及其主要成分")
plt.legend()
plt.tight_layout()
plt.show()

# 3) ACF/PACF 图
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df["y"], lags=60, ax=ax[0], color=bright_colors[4])
ax[0].set_title("图2A:ACF(自相关)")
plot_pacf(df["y"], lags=60, ax=ax[1], method="ywm", color=bright_colors[5])
ax[1].set_title("图2B:PACF(偏自相关)")
plt.tight_layout()
plt.show()

# 4) LSTM 数据准备
feature_cols = ["y", "marketing", "holiday", "sin_w", "cos_w", "sin_y", "cos_y"]
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# 仅用训练集拟合 scaler
X_train_fit = df_train[feature_cols].values
scaler_x.fit(X_train_fit)
scaler_y.fit(df_train[["y"]].values)

def make_lstm_windows(df_full, W=60, H=30, feature_cols=feature_cols):
    X, Y, T = [], [], []
    values = df_full[feature_cols].values
    values_x = scaler_x.transform(values.copy())
    values_y = scaler_y.transform(df_full[["y"]].values.copy())

    for i in range(W, len(df_full) - H + 1):
        x_win = values_x[i-W:i, :]      # [W, F]
        y_h = values_y[i:i+H].flatten() # [H]
        X.append(x_win)
        Y.append(y_h)
        T.append(df_full["ds"].iloc[i:i+H].values)
    X = np.array(X) # [N, W, F]
    Y = np.array(Y) # [N, H]
    return X, Y, T

X_all, Y_all, T_all = make_lstm_windows(df, W=W, H=H, feature_cols=feature_cols)

# 窗口切分与 df 切分对齐:窗口起点 >= train_end 对应验证区等
# 计算每个窗口的预测起点索引
win_starts = np.arange(W, len(df)-H+1)
# 训练窗口:预测起点 < train_end
train_mask = win_starts < train_end
# 验证窗口:预测起点 >= train_end 且 < val_end
val_mask = (win_starts >= train_end) & (win_starts < val_end)
# 测试窗口:预测起点 >= val_end
test_mask = (win_starts >= val_end)

X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
X_va, Y_va = X_all[val_mask], Y_all[val_mask]
X_te, Y_te = X_all[test_mask], Y_all[test_mask]

class SeqDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_ds = SeqDataset(X_tr, Y_tr)
val_ds   = SeqDataset(X_va, Y_va)
test_ds  = SeqDataset(X_te, Y_te)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

# 5) 定义 LSTM 模型:输入[W, F]，输出H步
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, horizon=30):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, horizon)
    def forward(self, x):
        # x: [B, W, F]
        out, (hn, cn) = self.lstm(x)
        h_last = out[:, -1, :]  # [B, hidden]
        y_hat  = self.fc(h_last) # [B, H]
        return y_hat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_lstm = LSTMForecaster(input_dim=len(feature_cols), hidden_dim=96, num_layers=2, dropout=0.2, horizon=H).to(device)
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 6) 训练 LSTM(早停)
best_val = np.inf
patience, wait = 8, 0
for epoch in range(80):
    model_lstm.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model_lstm(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_ds)

    # 验证
    model_lstm.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model_lstm(xb)
            val_loss += criterion(pred, yb).item() * xb.size(0)
    val_loss /= len(val_ds)

    print(f"[Epoch {epoch+1:03d}] train={train_loss:.4f} val={val_loss:.4f}")
    if val_loss < best_val - 1e-5:
        best_val = val_loss
        wait = 0
        best_state = {k: v.cpu().clone() for k, v in model_lstm.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

model_lstm.load_state_dict(best_state)
model_lstm.to(device)

# 7) XGBoost 训练(1步预测，递归多步)
def build_lag_features(df_in, lags=30, roll_list=[3,7,14]):
    df_tmp = df_in.copy()
    for L in range(1, lags+1):
        df_tmp[f"lag_{L}"] = df_tmp["y"].shift(L)
    for w in roll_list:
        df_tmp[f"roll_mean_{w}"] = df_tmp["y"].rolling(w).mean().shift(1)
        df_tmp[f"roll_std_{w}"] = df_tmp["y"].rolling(w).std().shift(1)
    # 外生与季节特征
    ex_cols = ["marketing", "holiday", "sin_w", "cos_w", "sin_y", "cos_y"]
    return df_tmp, ex_cols

lags = 30
df_feat, ex_cols = build_lag_features(df, lags=lags)
feature_cols_xgb = [c for c in df_feat.columns if c.startswith("lag_") or c.startswith("roll_")] + ex_cols

# 切除缺失(滞后导致前段NaN)
df_feat = df_feat.dropna().reset_index(drop=True)

# 划分 xgb 的训练、验证、测试索引
# 重新对齐切分点
cut_train = df_feat[df_feat["ds"] < df["ds"].iloc[train_end]].index.max()
cut_val   = df_feat[df_feat["ds"] < df["ds"].iloc[val_end]].index.max()

X_xgb = df_feat[feature_cols_xgb]
y_xgb = df_feat["y"]

X_tr_xgb = X_xgb.iloc[:cut_train+1]
y_tr_xgb = y_xgb.iloc[:cut_train+1]

X_va_xgb = X_xgb.iloc[cut_train+1:cut_val+1]
y_va_xgb = y_xgb.iloc[cut_train+1:cut_val+1]

dtrain = xgb.DMatrix(X_tr_xgb, label=y_tr_xgb)
dval   = xgb.DMatrix(X_va_xgb, label=y_va_xgb)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

xgb_model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dtrain, "train"), (dval, "val")],
                      early_stopping_rounds=50, verbose_eval=50)

# 8) SARIMAX 训练(使用周季节示例)
sarimax_order = (1,1,1)
seasonal_order = (1,1,1,7)
sarimax_model = SARIMAX(df_train["y"], order=sarimax_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
sarimax_res = sarimax_model.fit(disp=False)

# 9) Prophet 训练(加上营销作为回归项)
df_prophet_train = df_train[["ds", "y", "marketing"]].copy()
m = Prophet(seasonality_mode="additive", yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m.add_regressor("marketing")
m.fit(df_prophet_train.rename(columns={"ds": "ds", "y": "y"}))

# 10) 多步预测函数
def lstm_forecast(last_block_df, W=60, H=30):
    # 使用最后W天的特征窗口作为输入
    vals = last_block_df[feature_cols].values
    x = scaler_x.transform(vals[-W:])
    xt = torch.tensor(x[np.newaxis, :, :], dtype=torch.float32).to(device)
    model_lstm.eval()
    with torch.no_grad():
        y_h_pred_scaled = model_lstm(xt).cpu().numpy().ravel()
    y_h_pred = scaler_y.inverse_transform(y_h_pred_scaled.reshape(-1,1)).ravel()
    return y_h_pred

def xgb_recursive_forecast(df_hist, H=30, lags=30):
    df_tmp = df_hist.copy()
    preds = []
    for h in range(H):
        # 每一步都构造一行最新特征
        # 需要最近的滞后和滚动统计
        row = {}
        y_hist = df_tmp["y"].values
        for L in range(1, lags+1):
            row[f"lag_{L}"] = y_hist[-L]
        for w in [3,7,14]:
            row[f"roll_mean_{w}"] = pd.Series(y_hist).rolling(w).mean().iloc[-1]
            row[f"roll_std_{w}"] = pd.Series(y_hist).rolling(w).std().iloc[-1]
        last = df_tmp.iloc[-1]
        for c in ["marketing", "holiday", "sin_w", "cos_w", "sin_y", "cos_y"]:
            # 使用最近一天的外生与季节特征的延续/复制(简化做法)
            row[c] = last[c]
        X_one = pd.DataFrame([row])[feature_cols_xgb]
        dp = xgb.DMatrix(X_one)
        yhat = xgb_model.predict(dp)[0]
        preds.append(yhat)

        # 将预测值回写，便于下一步递归
        # 同时向前推进日期特征(简化:只平移sin/cos/营销/节假日按最近的)
        next_day = df_tmp["ds"].iloc[-1] + pd.Timedelta(days=1)
        df_tmp = pd.concat([df_tmp, pd.DataFrame({
            "ds": [next_day],
            "y": [yhat],
            "marketing": [last["marketing"]],
            "holiday": [0.0],  # 未来默认无节日(可按需要替换)
            "sin_w": [np.sin(2*np.pi*((last["ds"]+pd.Timedelta(days=1)).dayofweek)/7)],
            "cos_w": [np.cos(2*np.pi*((last["ds"]+pd.Timedelta(days=1)).dayofweek)/7)],
            "sin_y": [np.sin(2*np.pi*((last["ds"]+pd.Timedelta(days=1)).dayofyear)/365.25)],
            "cos_y": [np.cos(2*np.pi*((last["ds"]+pd.Timedelta(days=1)).dayofyear)/365.25)]
        })], ignore_index=True)
    return np.array(preds)

def sarimax_forecast(res, steps=30):
    fc = res.get_forecast(steps=steps)
    return fc.predicted_mean.values

def prophet_forecast(model, last_date, H=30, last_marketing=0.0):
    future = pd.DataFrame({"ds": pd.date_range(start=last_date+pd.Timedelta(days=1), periods=H, freq="D")})
    # 简化:未来营销使用最后一天的水平(或0)
    future["marketing"] = last_marketing
    forecast = model.predict(future)
    return forecast["yhat"].values, forecast["yhat_lower"].values, forecast["yhat_upper"].values, future["ds"].values

# 11) 在验证集末点进行一次H步预测，做权重寻优
last_block_trainval = df.iloc[:val_end].copy()
y_true_val_h = df["y"].iloc[val_end:val_end+H].values
val_dates = df["ds"].iloc[val_end:val_end+H].values

yhat_lstm_val = lstm_forecast(last_block_trainval, W=W, H=H)
yhat_xgb_val  = xgb_recursive_forecast(last_block_trainval, H=H, lags=lags)
sarimax_res_val = SARIMAX(df.iloc[:val_end]["y"], order=sarimax_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
yhat_arima_val = sarimax_forecast(sarimax_res_val, steps=H)
m_val = Prophet(seasonality_mode="additive", yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m_val.add_regressor("marketing")
m_val.fit(df.iloc[:val_end][["ds", "y", "marketing"]].rename(columns={"ds":"ds","y":"y"}))
yhat_prophet_val, low_p, up_p, _ = prophet_forecast(m_val, df.iloc[val_end-1]["ds"], H=H, last_marketing=df.iloc[val_end-1]["marketing"])

preds_val = np.vstack([yhat_xgb_val, yhat_lstm_val, yhat_arima_val, yhat_prophet_val]).T  # [H, 4]
y_true_v = y_true_val_h

def rmse(y, yhat):
    return sqrt(mean_squared_error(y, yhat))
def mape(y, yhat, eps=1e-6):
    return np.mean(np.abs((y - yhat) / (np.abs(y) + eps)))

# 粗粒度网格搜索权重(非负且和为1)
weights = []
errs = []
grid = np.linspace(0,1,11)
for w1 in grid:
    for w2 in grid:
        for w3 in grid:
            w4 = 1 - w1 - w2 - w3
            if w4 < 0: 
                continue
            w = np.array([w1,w2,w3,w4])
            y_ens = preds_val @ w
            e = rmse(y_true_v, y_ens)
            weights.append(w)
            errs.append(e)
best_w = weights[int(np.argmin(errs))]
print("Best weights on val (XGB, LSTM, ARIMA, Prophet)=", best_w, "RMSE=", np.min(errs))

# 12) 在测试集开始点进行一次H步预测，并作图比较
last_block_trainval2 = df.iloc[:val_end].copy()
y_true_test_h = df["y"].iloc[val_end:val_end+H].values
test_dates = df["ds"].iloc[val_end:val_end+H].values

yhat_lstm_te = lstm_forecast(last_block_trainval2, W=W, H=H)
yhat_xgb_te  = xgb_recursive_forecast(last_block_trainval2, H=H, lags=lags)
sarimax_res_te = SARIMAX(df.iloc[:val_end]["y"], order=sarimax_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
yhat_arima_te = sarimax_forecast(sarimax_res_te, steps=H)
m_te = Prophet(seasonality_mode="additive", yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m_te.add_regressor("marketing")
m_te.fit(df.iloc[:val_end][["ds","y","marketing"]].rename(columns={"ds":"ds","y":"y"}))
yhat_prophet_te, low_p_te, up_p_te, _ = prophet_forecast(m_te, df.iloc[val_end-1]["ds"], H=H, last_marketing=df.iloc[val_end-1]["marketing"])

preds_test = np.vstack([yhat_xgb_te, yhat_lstm_te, yhat_arima_te, yhat_prophet_te]).T
y_ens_te = preds_test @ best_w

# 估计简单不确定性带(基于验证残差的方差)
res_val = y_true_v - (preds_val @ best_w)
sigma = np.std(res_val)
lower_ens = y_ens_te - 1.96 * sigma
upper_ens = y_ens_te + 1.96 * sigma

# 13) 图3:XGBoost 特征重要性(彩色柱状图)
score_dict = xgb_model.get_score(importance_type="gain")
imp_df = pd.DataFrame({"feature": list(score_dict.keys()), "gain": list(score_dict.values())})
imp_df = imp_df.sort_values("gain", ascending=False).head(20)

plt.figure(figsize=(10, 6))
bars = plt.barh(imp_df["feature"], imp_df["gain"], color=sns.color_palette("husl", len(imp_df)))
plt.gca().invert_yaxis()
plt.title("图3:XGBoost 特征重要性(前20项)")
plt.xlabel("Gain")
plt.tight_layout()
plt.show()

# 14) 图4:预测对比(测试集起点的未来H步)
plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_true_test_h, color=bright_colors[1], lw=2.0, label="真实值")
plt.plot(test_dates, yhat_xgb_te, color=bright_colors[0], lw=1.5, label="XGBoost")
plt.plot(test_dates, yhat_lstm_te, color=bright_colors[2], lw=1.5, label="LSTM")
plt.plot(test_dates, yhat_arima_te, color=bright_colors[3], lw=1.5, label="ARIMA")
plt.plot(test_dates, yhat_prophet_te, color=bright_colors[4], lw=1.5, label="Prophet")
plt.plot(test_dates, y_ens_te, color=bright_colors[6], lw=2.5, label="融合(加权)")
plt.fill_between(test_dates, lower_ens, upper_ens, color=bright_colors[6], alpha=0.15, label="融合 95%区间")
plt.title("图4:多模型与融合预测对比(测试集前H步)")
plt.legend()
plt.tight_layout()
plt.show()

# 15) 图5:多步预测误差热图(模型×预测步数)
# 用rolling origins评估每个horizon的RMSE；为了示意，选用若干起点
def rolling_origin_errors(df_full, start_idx, end_idx, H=30, stride=10):
    models = ["XGB", "LSTM", "ARIMA", "Prophet"]
    errs = {m: [] for m in models}
    horizons = np.arange(1, H+1)
    for origin in range(start_idx, end_idx-H, stride):
        block = df_full.iloc[:origin].copy()
        y_true = df_full["y"].iloc[origin:origin+H].values

        y_xgb = xgb_recursive_forecast(block, H=H, lags=lags)
        y_lstm = lstm_forecast(block, W=W, H=H)
        sar = SARIMAX(df_full["y"].iloc[:origin], order=sarimax_order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        y_arima = sarimax_forecast(sar, steps=H)
        mp = Prophet(seasonality_mode="additive", yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        mp.add_regressor("marketing")
        mp.fit(df_full.iloc[:origin][["ds","y","marketing"]].rename(columns={"ds":"ds","y":"y"}))
        y_prophet, _, _, _ = prophet_forecast(mp, df_full.iloc[origin-1]["ds"], H=H, last_marketing=df_full.iloc[origin-1]["marketing"])

        for m, yh in zip(models, [y_xgb, y_lstm, y_arima, y_prophet]):
            errs[m].append((y_true - yh)**2)
    # 汇总到RMSE per horizon
    rmse_mat = np.zeros((len(models), H))
    for i, m in enumerate(models):
        if len(errs[m]) == 0:
            continue
        se = np.stack(errs[m], axis=0)   # [num_origins, H]
        rmse_mat[i] = np.sqrt(np.mean(se, axis=0))
    return models, horizons, rmse_mat

models, horizons, rmse_mat = rolling_origin_errors(df, start_idx=val_end, end_idx=min(val_end+120, len(df)-H-1), H=H, stride=10)

plt.figure(figsize=(10, 5))
sns.heatmap(rmse_mat, annot=False, cmap="magma", cbar=True, xticklabels=horizons, yticklabels=models)
plt.title("图5:多步预测RMSE热图(模型×预测步数)")
plt.xlabel("预测步(Horizon)")
plt.ylabel("模型")
plt.tight_layout()
plt.show()

# 16) 指标打印
def print_metrics(name, y_true, y_pred):
    print(f"{name}: RMSE={rmse(y_true, y_pred):.3f}, MAE={mean_absolute_error(y_true, y_pred):.3f}, MAPE={mape(y_true, y_pred)*100:.2f}%")

print_metrics("XGBoost(Test H)", y_true_test_h, yhat_xgb_te)
print_metrics("LSTM(Test H)", y_true_test_h, yhat_lstm_te)
print_metrics("ARIMA(Test H)", y_true_test_h, yhat_arima_te)
print_metrics("Prophet(Test H)", y_true_test_h, yhat_prophet_te)
print_metrics("Ensemble(Test H)", y_true_test_h, y_ens_te)
