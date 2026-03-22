#!/bin/bash
# 先杀掉同名 tmux 会话，避免重复
SESSION="coin_collector"
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Python 可执行文件路径
PY_ENV="/root/miniconda3/envs/rapids-25.08/bin/python3"

# 具体要执行的命令行
CMD="$PY_ENV /root/okex_websocket.py stream \
  --inst BTC-USDT ETH-USDT XRP-USDT SOL-USDT BNB-USDT USDC-USDT \
         BTC-USDT-SWAP ETH-USDT-SWAP XRP-USDT-SWAP BNB-USDT-SWAP SOL-USDT-SWAP \
  --depth-channel books5 \
  --candle 1m \
  --candle-mode final \
  --ticks \
  --format feather \
  --out /root/okx_data \
  --log-level INFO"

# 创建新的 tmux 会话执行命令，结束后保留一个 bash
tmux new-session -d -s "$SESSION" "$CMD ; bash"
