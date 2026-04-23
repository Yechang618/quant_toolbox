#!/bin/bash

# ==========================================
# 脚本名称: run_factor_gen.sh
# 功能: 在服务器上使用 nohup 后台运行特征生成脚本
# ==========================================

# 1. 确保工作目录为脚本所在目录（通常为项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "❌ 无法切换到项目目录"; exit 1; }

# 2. 配置虚拟环境（按需取消注释并修改路径）
# VENV_DIR="./.venv2"
# if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
#     source "$VENV_DIR/bin/activate"
#     echo "✅ 已激活虚拟环境: $VENV_DIR"
# else
#     echo "⚠️ 未找到虚拟环境，将使用系统 Python"
# fi

# 3. 配置运行参数（参考 python run_factor_gen.py --help）
# 示例：--symbol BTCUSDT --symbol ETHUSDT --dry-run
EXTRA_ARGS="" 

# 4. 日志配置（按时间戳自动生成日志文件）
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_factor_gen_$(date +%Y%m%d_%H%M%S).log"

# 5. 执行命令
PYTHON="python3"  # 若服务器默认 python 指向 python2，请改为 python3 或绝对路径

echo "🚀 正在启动 run_factor_gen.py ..."
echo "📄 日志输出路径: $LOG_FILE"
echo "🔧 附加参数: ${EXTRA_ARGS:-无}"

# 使用 nohup 后台运行，合并标准输出与错误输出
# nohup $PYTHON run_factor_gen.py $EXTRA_ARGS > "$LOG_FILE" 2>&1 &
nohup $PYTHON ./script/run_factor_gen.py $EXTRA_ARGS > "$LOG_FILE" 2>&1 &

# 记录进程 PID 便于后续管理
echo $! > run_factor_gen.pid
echo "✅ 进程已启动，PID: $!"
echo "💡 实时查看日志: tail -f $LOG_FILE"
echo "💡 停止任务:   kill \$(cat run_factor_gen.pid)"