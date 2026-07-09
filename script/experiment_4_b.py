#!/usr/bin/env python3
"""
script/experiment_4.py
网格搜索最优 (b, alpha) 参数组合，计算 Method 1 混合因子的 IC 值与 MAD 值。
公式: X_mix = [ Σ(exp(b·x1)·x1) + α·Σ(exp(b·x2)·x2) ] / [ Σ(exp(b·x1)) + α·Σ(exp(b·x2)) ] - threshold
✅ 核心特性：
  1. 动态计算权重：严格根据 w_{mi} = exp(b·x_{mi})/Σexp(b·x_{mi}) 实时计算
  2. 数值稳定：采用 log-shift 技巧防止 exp(±1000·x) 溢出
  3. 双指标评估：同时输出 IC (Spearman) 与 MAD (Mean Absolute Difference) 热力图
  4. 4组独立实验：按 basis_cols 定义生成 8 张热力图 (4 IC + 4 MAD)
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# from scipy.stats import spearmanr
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from util.config import settings
from util.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# 📥 数据解析与提取工具
# =============================================================================
def _parse_ob_window(data):
    """安全解析 Parquet 中的订单簿窗口数据，兼容 JSON字符串/List[Dict]/List[String]"""
    if data is None: return pd.DataFrame()
    if isinstance(data, str):
        data = data.strip()
        if not data or data == '[]': return pd.DataFrame()
        try: return pd.read_json(data, orient='records')
        except: return pd.DataFrame()
    if isinstance(data, (list, np.ndarray, pd.Series)):
        if len(data) == 0: return pd.DataFrame()
        parsed = []
        for item in data:
            if isinstance(item, str):
                try: parsed.append(json.loads(item))
                except: pass
            elif isinstance(item, dict):
                parsed.append(item)
        return pd.DataFrame(parsed) if parsed else pd.DataFrame()
    return pd.DataFrame()

def extract_basis_arrays(parquet_dir, mode_filter=None, op_filter=None):
    """从 Parquet 文件中提取所有样本的基础因子序列、标签与阈值"""
    logger.info(f"📥 从 {parquet_dir} 加载数据...")
    pq_files = sorted(Path(parquet_dir).glob("*_windows.parquet"))
    if not pq_files:
        raise FileNotFoundError(f"未找到 Parquet 文件: {parquet_dir}")
    
    Y, thresh, X_baa, X_bbb, X_bab, X_bba = [], [], [], [], [], []
    stats = {"total": 0, "skip_filter": 0, "skip_na": 0, "skip_empty": 0, "skip_cols": 0, "skip_short": 0, "valid": 0}
    required_cols = ['spot_bid1_px', 'spot_ask1_px', 'swap_bid1_px', 'swap_ask1_px']
    
    for pq in tqdm(pq_files, desc="路由样本"):
        df = pd.read_parquet(pq, engine='pyarrow')
        for _, row in df.iterrows():
            stats["total"] += 1
            rec = row.get('record', {})
            if not rec: 
                stats["skip_na"] += 1
                continue
                
            mode = rec.get('trade_mode')
            op = rec.get('operation')
            
            if mode_filter is not None and mode != mode_filter: 
                stats["skip_filter"] += 1
                continue
            if op_filter is not None and op != op_filter: 
                stats["skip_filter"] += 1
                continue
                
            y = rec.get('gain_vs_threshold')
            th = rec.get('threshold')
            if pd.isna(y) or pd.isna(th):
                stats["skip_na"] += 1
                continue

            ob = _parse_ob_window(row.get('ob_window'))
            if ob.empty:
                stats["skip_empty"] += 1
                continue
                
            if not all(c in ob.columns for c in required_cols):
                stats["skip_cols"] += 1
                continue

            valid = ob[required_cols].dropna()
            if len(valid) < 5:
                stats["skip_short"] += 1
                continue
            
            # 计算 4 类基础因子序列
            baa = np.log(valid['swap_ask1_px']) - np.log(valid['spot_ask1_px']) - th
            bbb = np.log(valid['swap_bid1_px']) - np.log(valid['spot_bid1_px']) - th
            bab = np.log(valid['swap_ask1_px']) - np.log(valid['spot_bid1_px']) - th
            bba = np.log(valid['swap_bid1_px']) - np.log(valid['spot_ask1_px']) - th
            
            min_len = min(len(baa), len(bbb), len(bab), len(bba))
            Y.append(y)
            thresh.append(th)
            X_baa.append(baa.iloc[:min_len].values)
            X_bbb.append(bbb.iloc[:min_len].values)
            X_bab.append(bab.iloc[:min_len].values)
            X_bba.append(bba.iloc[:min_len].values)
            stats["valid"] += 1

    logger.info(f"📊 样本路由统计: 总记录={stats['total']} | 过滤跳过={stats['skip_filter']} | 缺失/空值={stats['skip_na']+stats['skip_empty']} | 列缺失={stats['skip_cols']} | 窗口过短={stats['skip_short']} | ✅ 有效={stats['valid']}")
    return np.array(Y), np.array(thresh), X_baa, X_bbb, X_bab, X_bba

# =============================================================================
# 🧮 核心计算：网格 IC & MAD 评估
# =============================================================================
def compute_safe_weighted_sum(b, x):
    """数值稳定版：计算 Σ(exp(b·x)·x) 与 Σ(exp(b·x))，防止溢出"""
    if len(x) == 0: return 0.0, 0.0
    bx = b * x
    # shift = np.max(bx)
    shift = 0
    exp_bx = np.exp(bx - shift)
    return np.dot(exp_bx, x), exp_bx.sum()

def run_grid_search(b_vals, alpha, x1_list, x2_list, labels, thresholds, eta):
    """
    执行 (b1, b2) 网格搜索，返回 IC 矩阵与 MAD 矩阵
    eta: 1.0 (open2) 或 -1.0 (close2)，用于 MAD 符号调整
    """
    ic_matrix = np.full((len(b_vals), len(b_vals)), np.nan)
    mad_matrix = np.full((len(b_vals), len(b_vals)), np.nan)
    
    # 预过滤 NaN/短序列样本提升速度
    valid_mask = [len(x1) > 0 for x1 in x1_list]
    x1_v = [x for x, v in zip(x1_list, valid_mask) if v]
    x2_v = [x for x, v in zip(x2_list, valid_mask) if v]
    lab_v = labels[valid_mask]
    thr_v = thresholds[valid_mask]
    n_samples = len(lab_v)
    
    pbar = tqdm(total=len(b_vals) * len(b_vals), desc="网格计算 IC/MAD", leave=False)
    
    for i, b1 in enumerate(b_vals):
        for j, b2 in enumerate(b_vals):
            pred = np.full(n_samples, np.nan)
            for m in range(n_samples):
                s1_num, s1_den = compute_safe_weighted_sum(b1, x1_v[m])
                s2_num, s2_den = compute_safe_weighted_sum(b2, x2_v[m])
                den = s1_den + alpha * s2_den
                if den > 1e-12:
                    pred[m] = (s1_num + alpha * s2_num) / den - thr_v[m]
                    
            valid_pred = ~np.isnan(pred)
            n_valid = valid_pred.sum()
            
            if n_valid >= 10:
                # IC: Spearman 秩相关
                # => pearsonr, instead of spearmanr, for performance and stability
                rho, _ = pearsonr(lab_v[valid_pred], pred[valid_pred])
                ic_matrix[i, j] = rho if np.isfinite(rho) else np.nan
                
                # 🔥 MAD: Mean Absolute Difference，按 operation 适配符号
                # open2: eta=1 => |y - x|; close2: eta=-1 => |y + x| = |y - (-1)*x|
                abs_diff = np.abs(lab_v[valid_pred] - eta * pred[valid_pred])
                mad_matrix[i, j] = np.mean(abs_diff)
            pbar.update(1)
    pbar.close()
    return ic_matrix, mad_matrix

# =============================================================================
# 📊 绘图与输出
# =============================================================================
def plot_and_save_heatmap(matrix, b_vals, title, save_path, cmap='RdBu_r', center=0, vmin=None, vmax=None, cbar_label='Value'):
    """通用热力图绘制函数，支持 IC/MAD 不同配色"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 自动推断 colorbar 范围（若未指定）
    if vmin is None: vmin = np.nanmin(matrix)
    if vmax is None: vmax = np.nanmax(matrix)
    
    sns.heatmap(matrix, ax=ax, 
                xticklabels=np.round(b_vals, 2), 
                yticklabels=np.round(b_vals, 1),
                cmap=cmap, center=center, vmin=vmin, vmax=vmax,
                cbar_kws={'label': cbar_label})
    ax.set_xlabel('Weight Parameter (b1)')
    ax.set_ylabel('Weight Parameter (b2)')
    ax.set_title(title, fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 热力图已保存 → {save_path}")

# =============================================================================
# 🚀 主流程
# =============================================================================
def run_experiment_4_b(
    input_dir: Path = None,
    output_dir: Path = None,
    b_min: float = -1000.0, b_max: float = 1000.0, b_steps: int = 21,
    alpha: float = 0.1, 
    mode_filter: int = None,
    op_filter: str = None
):
    logger.info("=== 启动 Experiment 4 b: 网格搜索最优 b1 & b2 (IC + MAD) ===")
    if input_dir is None: input_dir = settings.output_root / "step1_windows"
    if output_dir is None: output_dir = settings.output_root / "experiment_4_b_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    Y, thresh, X_baa, X_bbb, X_bab, X_bba = extract_basis_arrays(input_dir, mode_filter, op_filter)
    if len(Y) < 50:
        raise ValueError(f"有效样本仅 {len(Y)}，无法进行网格评估。请检查数据过滤条件或放宽阈值。")

    # 2. 配置网格
    b_vals = np.linspace(b_min, b_max, b_steps)
    # alpha_vals = np.array([alpha])
    
    basis_map = {'X_baa': X_baa, 'X_bbb': X_bbb, 'X_bab': X_bab, 'X_bba': X_bba}
    basis_cols = {'baa': ['bba', 'bab'], 'bbb': ['bab', 'bba']}  # 4 种混合方案
    
    # 3. 遍历 4 种组合，分别计算 IC 与 MAD
    for b1, b2_list in basis_cols.items():
        for b2 in b2_list:
            combo_name = f"{b1}_{b2}"
            logger.info(f"\n🔍 评估组合: {combo_name}")
            
            x1 = basis_map[f'X_{b1}']
            x2 = basis_map[f'X_{b2}']
            
            # 🔥 核心：传入 eta 用于 MAD 符号计算
            eta = 1.0 if op_filter == 'open2' else -1.0
            ic_mat, mad_mat = run_grid_search(b_vals, alpha, x1, x2, Y, thresh, eta)
            
            # 📊 保存 IC 热力图 (RdBu_r: 红负蓝正，中心 0)
            ic_path = output_dir / f"ichm_alpha_{alpha}_{combo_name}_range{b_min}_{b_max}_mod{mode_filter}_{op_filter}.png"
            plot_and_save_heatmap(ic_mat, b_vals, 
                                  f'IC Heatmap: BWT_{b1}_{b2}_{alpha} (Method 1)', 
                                  ic_path, cmap='RdBu_r', center=0, vmin=-1, vmax=1, cbar_label='Spearman IC')
            
            # 📊 保存 MAD 热力图 (viridis: 越小越好，低值深色)
            mad_path = output_dir / f"mad_heatmap_alpha_{alpha}_{combo_name}_range{b_min}_{b_max}_mod{mode_filter}_{op_filter}.png"
            # MAD 范围自适应，通常 0~2 之间
            mad_vmax = np.nanpercentile(mad_mat, 95) if np.any(np.isfinite(mad_mat)) else 1.0
            plot_and_save_heatmap(mad_mat, b_vals, 
                                  f'MAD Heatmap: BWT_{b1}_{b2}_{alpha} (Method 1)', 
                                  mad_path, cmap='viridis_r', center=None, vmin=0, vmax=mad_vmax, cbar_label='Mean |y - η·x|')
            
            # 🏆 打印最优结果（IC 取绝对值最大，MAD 取最小）
            if np.any(np.isfinite(ic_mat)):
                idx_ic = np.unravel_index(np.nanargmax(np.abs(ic_mat)), ic_mat.shape)
                logger.info(f"🏆 IC 最优: |IC|={np.abs(ic_mat[idx_ic]):.4f} | b1={b_vals[idx_ic[0]]:.1f} | b2={b_vals[idx_ic[1]]:.1f}")
            if np.any(np.isfinite(mad_mat)):
                idx_mad = np.unravel_index(np.nanargmin(mad_mat), mad_mat.shape)
                logger.info(f"🏆 MAD 最优: MAD={mad_mat[idx_mad]:.4f} | b1={b_vals[idx_mad[0]]:.1f} | b2={b_vals[idx_mad[1]]:.1f}")

    logger.info(f"\n✅ 全部热力图已保存至: {output_dir}")
    logger.info("=== Experiment 4 完成 ===")

if __name__ == "__main__":
    run_experiment_4_b(
        input_dir=Path("data_processed/step1_windows_eq_tick"),
        output_dir=Path("output/experiment_4_b_heatmaps"),
        b_min=-10000, b_max=10000, b_steps=21,
        alpha = 0.5,
        # 按需修改为 0, 2 或 None
        # ✅ 核心：'open2' 或 'close2'，用于 MAD 符号计算
        # mode_filter=0, op_filter='open2'          
        mode_filter=2, op_filter='open2' 
        # mode_filter=0, op_filter='close2' 
        # mode_filter=2, op_filter='close2' 
    )