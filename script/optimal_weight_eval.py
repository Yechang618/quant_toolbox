#!/usr/bin/env python3
"""
optimal_weight_eval.py
评估并寻找最优全局权重参数 b，使得 L1 损失最小化：
L(b) = sum_m | y_m - eta_m * sum_i( w_{mi} * x_{mi} ) |
其中 w_{mi} = exp(b * x_{mi}) / sum_i(exp(b * x_{mi}))

✅ 核心更新：按 mode(0/2) 与 operation(open2/close2) 拆分为 4 组独立实验
✅ 保留修复：安全解析 Parquet 嵌套数据、变长窗口兼容、Softmax 数值稳定、NaN 自动过滤
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from util.config import settings
from util.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# 📥 安全数据解析器（彻底修复 ndarray 布尔歧义）
# =============================================================================
def _parse_ob_window(data):
    if data is None:
        return pd.DataFrame()
    if isinstance(data, str):
        data = data.strip()
        if not data or data == '[]':
            return pd.DataFrame()
        try:
            return pd.read_json(data, orient='records')
        except Exception:
            return pd.DataFrame()
    if isinstance(data, (list, np.ndarray, pd.Series)):
        if len(data) == 0:
            return pd.DataFrame()
        parsed = []
        for item in data:
            if isinstance(item, str):
                try: parsed.append(json.loads(item))
                except: pass
            elif isinstance(item, dict):
                parsed.append(item)
        return pd.DataFrame(parsed) if parsed else pd.DataFrame()
    return pd.DataFrame()

# =============================================================================
# 🧮 核心数学与优化逻辑
# =============================================================================
def compute_l1_loss(b, X_list, Y, eta):
    """
    计算给定 b 的 L1 损失（逐样本计算，天然支持变长窗口）
    eta: 标量，1.0 (open2) 或 -1.0 (close2)
    """
    total_loss = 0.0
    for m in range(len(Y)):
        x_m = X_list[m]
        if len(x_m) == 0: continue
        
        # 数值稳定版 Softmax 加权
        if np.abs(b) < 1e-12:
            pred = np.mean(x_m)
        else:
            bx = b * x_m
            bx_max = np.max(bx)
            exp_bx = np.exp(bx - bx_max)
            pred = np.sum(exp_bx * x_m) / np.sum(exp_bx)
            
        total_loss += np.abs(Y[m] - eta * pred)
    return total_loss

def find_optimal_b(X_list, Y, eta, b_range=(-10000.0, 10000.0), steps=100):
    """网格搜索 + 局部精细优化寻找最优 b"""
    bs = np.linspace(b_range[0], b_range[1], steps)
    losses = np.array([compute_l1_loss(b, X_list, Y, eta) for b in bs])
    
    idx_min = np.argmin(losses)
    b0 = bs[idx_min]
    loss0 = losses[idx_min]
    
    try:
        res = minimize_scalar(
            lambda b: compute_l1_loss(b, X_list, Y, eta),
            bounds=(max(b_range[0], b0 - 1.0), min(b_range[1], b0 + 1.0)),
            method='bounded',
            options={'xatol': 1e-4}
        )
        print(f"🔍 局部优化结果: b_opt={res.x:.4f}, loss={res.fun:.4f}")
        return res.x, res.fun, bs, losses
    except Exception:
        print("⚠️ 局部优化失败，返回网格搜索结果")
        return b0, loss0, bs, losses

# =============================================================================
# 🚀 主流程：4 组独立实验
# =============================================================================
def run_optimal_weight_evaluation(
    input_dir: Path = None,
    output_dir: Path = None,
    factor_types: list = ['baa', 'bbb', 'bab', 'bba'],
    min_window_len: int = 5,
    debug: bool = False
):
    logger.info("=== 启动最优加权参数 b 评估 (4组独立实验) ===")
    if input_dir is None:
        input_dir = settings.output_root / "step1_windows"
    if output_dir is None:
        output_dir = settings.output_root / "weight_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定义 4 个实验组
    groups = [(0, 'open2'), (0, 'close2'), (2, 'open2'), (2, 'close2')]
    group_data = {g: {'Y': [], 'X': {ft: [] for ft in factor_types}, 'count': 0} for g in groups}
    stats = {"total": 0, "skipped": 0}

    parquet_files = sorted(input_dir.glob("*_windows.parquet"))
    logger.info(f"共找到 {len(parquet_files)} 个 Parquet 文件，开始路由分组...")

    for pq_path in tqdm(parquet_files, desc="提取并路由样本"):
        df_windows = pd.read_parquet(pq_path, engine='pyarrow')
        stats["total"] += len(df_windows)
        
        for _, row in df_windows.iterrows():
            rec = row.get('record', {})
            if not rec: 
                stats["skipped"] += 1
                continue
                
            mode = rec.get('trade_mode')
            op = rec.get('operation')
            key = (mode, op)
            
            # 仅处理目标 4 组
            if key not in group_data:
                stats["skipped"] += 1
                continue
                
            y_m = rec.get('gain_vs_threshold')
            thres = rec.get('threshold')
            if pd.isna(y_m) or pd.isna(thres):
                stats["skipped"] += 1
                continue

            ob_df = _parse_ob_window(row.get('ob_window'))
            if ob_df.empty:
                stats["skipped"] += 1
                continue
                
            cols = ['spot_bid1_px', 'spot_ask1_px', 'swap_bid1_px', 'swap_ask1_px']
            valid = ob_df[cols].dropna()
            if len(valid) < min_window_len:
                stats["skipped"] += 1
                continue
                
            # 严格按公式计算 4 类基础因子序列 (x_{mi})
            basis_baa = np.log(valid['swap_ask1_px']) - np.log(valid['spot_ask1_px']) - thres
            basis_bbb = np.log(valid['swap_bid1_px']) - np.log(valid['spot_bid1_px']) - thres
            basis_bab = np.log(valid['swap_ask1_px']) - np.log(valid['spot_bid1_px']) - thres
            basis_bba = np.log(valid['swap_bid1_px']) - np.log(valid['spot_ask1_px']) - thres
            
            min_len = min(len(basis_baa), len(basis_bbb), len(basis_bab), len(basis_bba))
            
            # 路由至对应组
            group_data[key]['Y'].append(y_m)
            group_data[key]['X']['baa'].append(basis_baa.iloc[:min_len].values)
            group_data[key]['X']['bbb'].append(basis_bbb.iloc[:min_len].values)
            group_data[key]['X']['bab'].append(basis_bab.iloc[:min_len].values)
            group_data[key]['X']['bba'].append(basis_bba.iloc[:min_len].values)
            group_data[key]['count'] += 1

    logger.info(f"📊 路由统计: 总记录={stats['total']} | 成功分组={sum(g['count'] for g in group_data.values())} | 跳过={stats['skipped']}")

    # =====================================================================
    # 逐组优化与输出
    # =====================================================================
    all_results = {}
    
    for mode, op in groups:
        group_key = (mode, op)
        logger.info(f"\n{'='*40}")
        logger.info(f"🔍 开始实验组: mode={mode}, operation={op} | 样本数: {group_data[group_key]['count']}")
        logger.info(f"{'='*40}")
        
        if group_data[group_key]['count'] < 50:
            logger.warning(f"⚠️ 样本过少，跳过该组优化")
            continue
            
        Y = np.array(group_data[group_key]['Y'])
        eta = 1.0 if op == 'open2' else -1.0
        
        group_results = {}
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, ft in enumerate(factor_types):
            X_list = group_data[group_key]['X'][ft]
            b_opt, loss_min, bs_grid, losses_grid = find_optimal_b(X_list, Y, eta)
            
            group_results[ft] = {
                'b_opt': b_opt, 
                'min_loss': loss_min, 
                'n_samples': len(X_list),
                'eta': eta
            }
            logger.info(f"✅ {ft.upper()}: b_opt = {b_opt:.4f} | L1 Loss = {loss_min:.4f} | η = {eta}")
            
            # 绘图
            axes[idx].plot(bs_grid, losses_grid/len(X_list), lw=2, label='L1 Loss')
            axes[idx].axvline(b_opt, color='r', linestyle='--', lw=2, label=f'b*={b_opt:.3f}')
            axes[idx].set_title(f'Factor: {ft.upper()} (N={len(X_list)}, η={int(eta)})', fontweight='bold')
            axes[idx].set_xlabel('Weight Parameter b')
            axes[idx].set_ylabel('Total L1 Loss')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / f"loss_curves_nSample{group_data[group_key]['count']}_mode{mode}_{op}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 损失曲线已保存 → {plot_path.name}")

        # 保存该组结果
        res_df = pd.DataFrame(group_results).T
        res_df.index.name = 'factor_type'
        csv_path = output_dir / f"results_nSample{group_data[group_key]['count']}_mode{mode}_{op}.csv"
        res_df.to_csv(csv_path)
        logger.info(f"📄 评估结果已保存 → {csv_path.name}")
        
        all_results[f"mode{mode}_{op}"] = group_results

    # 汇总打印
    logger.info("\n🏁 === 全部实验完成 ===")
    for g_name, res in all_results.items():
        logger.info(f"  [{g_name}] baa:{res['baa']['b_opt']:.3f} | bbb:{res['bbb']['b_opt']:.3f} | bab:{res['bab']['b_opt']:.3f} | bba:{res['bba']['b_opt']:.3f}")
        
    return all_results

if __name__ == "__main__":
    run_optimal_weight_evaluation(
        input_dir=Path("data/step1_windows"),
        output_dir=Path("output/weight_evaluation"),
        factor_types=['baa', 'bbb', 'bab', 'bba'],
        min_window_len=5,
        debug=True
    )