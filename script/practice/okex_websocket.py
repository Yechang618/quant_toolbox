#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OKX WebSocket streamer + partitioned-dataset utilities (stable)

增强点
- 三连接（Depth/Public, Candle/Business, Trades/Public），逐条订阅
- 稳定性：更清晰的错误日志、带抖动的指数回退重连、订阅发送保护
- K线 final 模式增加超时定稿：--finalize-grace-mult (默认2x周期)
- 所有数据新增 ts_iso（UTC）= 'YYYY-MM-DD HH:MM:SS.mmm'
- 存储：
  1) 普通模式(未指定 --dataset)：按日单文件覆盖写 (feather|parquet)
  2) 分区数据集(--dataset parquet)：Hive 年/月/日 追加写（真·append）
- 工具：copy-range / read-range

依赖:
    pip install websocket-client pandas pyarrow
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable, Tuple, Iterable

import pandas as pd
from websocket import WebSocketApp

DEFAULT_WS_PUBLIC   = "wss://ws.okx.com:8443/ws/v5/public"
DEFAULT_WS_BUSINESS = "wss://ws.okx.com:8443/ws/v5/business"

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
except Exception:
    pa = pq = ds = None

# ----------------------------- Utils ----------------------------- #

def utc_ms_now() -> int:
    return int(time.time() * 1000)

def ts_ms_to_iso(ts_ms: int) -> str:
    """UTC -> 'YYYY-MM-DD HH:MM:SS.mmm'"""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    # keep 3 decimals for ms
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(dt.microsecond/1000):03d}"

def ymd_utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y%m%d")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_candle_choice(c: str) -> str:
    c = c.strip()
    lc = c.lower()
    valid = {"1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"}
    if lc not in valid:
        raise ValueError(f"Unsupported candle interval: {c}. Choose from 1m/3m/5m/15m/30m/1H/4H/1D")
    if lc.endswith("m"):
        return f"candle{lc}"
    if lc.endswith("h"):
        return f"candle{lc[:-1].upper()}H"
    if lc.endswith("d"):
        return f"candle{lc[:-1].upper()}D"
    raise ValueError(f"Invalid candle spec: {c}")

def candle_period_ms(candle_channel: str) -> int:
    # candle1m/3m/5m/15m/30m/1H/4H/1D
    s = candle_channel.replace("candle", "")
    if s.endswith("m"):
        return int(s[:-1]) * 60_000
    if s.endswith("H"):
        return int(s[:-1]) * 60 * 60_000
    if s.endswith("D"):
        return int(s[:-1]) * 24 * 60 * 60_000
    # fallback 1m
    return 60_000

def iter_dates(start_date: str, end_date: str) -> Iterable[datetime]:
    s = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    e = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if s > e:
        raise ValueError("start date > end date")
    cur = s
    one = timedelta(days=1)
    while cur <= e:
        yield cur
        cur += one

def ymd_parts_from_date(dt: datetime) -> Tuple[int, str, str]:
    return dt.year, f"{dt.month:02d}", f"{dt.day:02d}"

# ----------------------------- Storage ----------------------------- #

class FileSink:
    """
    普通模式：<out>/<bucket>/<YYYYMMDD>.(feather|parquet) 覆盖写
    数据集模式(--dataset parquet)：<out>/<bucket>/year=YYYY/month=MM/day=DD/part-*.parquet 追加写
    所有记录都会额外包含 ts_iso（UTC）
    """
    def __init__(self,
                 out_root: str,
                 file_format: str = "feather",
                 dataset: Optional[str] = None,
                 chunk_rows: int = 2000,
                 flush_secs: int = 10) -> None:
        if dataset is not None and dataset != "parquet":
            raise ValueError("Only 'parquet' dataset is supported")
        if dataset == "parquet" and pa is None:
            raise RuntimeError("pyarrow is required for dataset mode. pip install pyarrow")
        if file_format not in ("feather", "parquet"):
            raise ValueError("file_format must be 'feather' or 'parquet'")

        self.out_root = out_root
        self.file_format = file_format
        self.dataset = dataset
        self.chunk_rows = chunk_rows
        self.flush_secs = flush_secs

        self._buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        # 关键修复：按 bucket 记录上次 flush 时间，避免高频 bucket 刷新挤占低频 bucket
        self._last_flush: Dict[str, float] = {}
        self._totals: Dict[Tuple[str, str], pd.DataFrame] = {}

    def _yyyy_mm_dd_from_ts(self, ts_ms: int) -> Tuple[str, str, str]:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return str(dt.year), f"{dt.month:02d}", f"{dt.day:02d}"

    def add(self, bucket: str, record: Dict[str, Any]) -> None:
        # 确保 ts_iso 存在
        ts_ms = int(record.get("ts", utc_ms_now()))
        record.setdefault("ts", ts_ms)
        record.setdefault("ts_iso", ts_ms_to_iso(ts_ms))

        with self._lock:
            buf = self._buffers.setdefault(bucket, [])
            buf.append(record)
            now = time.time()
            last = self._last_flush.get(bucket, 0.0)
            if len(buf) >= self.chunk_rows or (now - last) >= self.flush_secs:
                self._flush_bucket_locked(bucket)
                self._last_flush[bucket] = now

    def flush_all(self) -> None:
        with self._lock:
            for bucket in list(self._buffers.keys()):
                self._flush_bucket_locked(bucket)
            if self.dataset is None:
                for key in list(self._totals.keys()):
                    self._write_daily_file_locked(key)

    def _flush_bucket_locked(self, bucket: str) -> None:
        buf = self._buffers.get(bucket)
        if not buf:
            return
        df_batch = pd.DataFrame(buf)
        self._buffers[bucket] = []
        if df_batch.empty:
            return

        if self.dataset == "parquet":
            self._append_dataset_batch(bucket, df_batch)
        else:
            df_batch["YYYYMMDD"] = df_batch["ts"].astype("int64").apply(ymd_utc)
            for day, df_day in df_batch.groupby("YYYYMMDD"):
                key = (bucket, day)
                df_day = df_day.drop(columns=["YYYYMMDD"])
                if key in self._totals:
                    self._totals[key] = pd.concat([self._totals[key], df_day], ignore_index=True)
                else:
                    self._totals[key] = df_day
                self._write_daily_file_locked(key)

    def _write_daily_file_locked(self, key: Tuple[str, str]) -> None:
        bucket, day = key
        out_dir = os.path.join(self.out_root, bucket)
        ensure_dir(out_dir)
        fpath = os.path.join(out_dir, f"{day}.{self.file_format}")
        df = self._totals.get(key)
        if df is None or df.empty:
            return
        try:
            if self.file_format == "feather":
                df.reset_index().to_feather(fpath)
            else:
                df.to_parquet(fpath, index=False)
        except Exception as e:
            print(f"[FileSink] ERROR writing {fpath}: {e}", file=sys.stderr)

    def _append_dataset_batch(self, bucket: str, df_batch: pd.DataFrame) -> None:
        if pa is None or ds is None:
            raise RuntimeError("pyarrow is required for dataset mode")

        years, months, days = [], [], []
        for ts_ms in df_batch["ts"].astype("int64").tolist():
            y, m, d = self._yyyy_mm_dd_from_ts(ts_ms)
            years.append(int(y)); months.append(m); days.append(d)

        df_batch = df_batch.copy()
        df_batch["year"] = years
        df_batch["month"] = months
        df_batch["day"] = days

        table = pa.Table.from_pandas(df_batch, preserve_index=False)
        base_dir = os.path.join(self.out_root, bucket)
        ensure_dir(base_dir)

        part_schema = pa.schema([
            pa.field("year", pa.int32()),
            pa.field("month", pa.string()),
            pa.field("day", pa.string()),
        ])

        try:
            ds.write_dataset(
                data=table,
                base_dir=base_dir,
                format="parquet",
                partitioning=ds.partitioning(part_schema, flavor="hive"),
                existing_data_behavior="overwrite_or_ignore",
            )
        except Exception as e:
            print(f"[FileSink] ERROR write_dataset to {base_dir}: {e}", file=sys.stderr)

# ------------------------ Candle Aggregator ------------------------ #

class CandleAggregator:
    """
    --candle-mode final:
    - 仅保留每根 K 线最后一个版本
    - 当发现新 ts 到来时，把旧 ts 定稿写入
    - 还支持“超时定稿”：超过 grace_ms 未更新则强制写出
    """
    def __init__(self, sink: FileSink, candle_channel: str,
                 period_ms: int, grace_mult: float = 2.0) -> None:
        self.sink = sink
        self.candle_channel = candle_channel
        self.period_ms = period_ms
        self.grace_ms = int(period_ms * grace_mult)
        self._cur: Dict[str, Dict[str, Any]] = {}           # instId -> bar dict
        self._last_recv_ms: Dict[str, int] = {}             # instId -> last receive wallclock (ms)
        self._lock = threading.Lock()

    def _bucket(self, inst: str) -> str:
        return f"candles/{self.candle_channel}/{inst}"

    def ingest(self, rec: Dict[str, Any]) -> None:
        inst = rec["instId"]
        ts = int(rec["ts"])
        now_ms = utc_ms_now()
        rec.setdefault("ts_iso", ts_ms_to_iso(ts))

        with self._lock:
            cur = self._cur.get(inst)
            self._last_recv_ms[inst] = now_ms
            if cur is None:
                self._cur[inst] = rec
                return
            if ts == int(cur["ts"]):
                self._cur[inst] = rec
                return
            # 新 ts 到来 -> 旧 ts 定稿
            self.sink.add(self._bucket(inst), cur)
            self._cur[inst] = rec

    def tick(self) -> None:
        """周期调用：检测超时未更新的bar，强制定稿"""
        now_ms = utc_ms_now()
        to_flush: List[Tuple[str, Dict[str, Any]]] = []
        with self._lock:
            for inst, cur in list(self._cur.items()):
                last_rcv = self._last_recv_ms.get(inst, 0)
                if last_rcv and (now_ms - last_rcv) >= self.grace_ms:
                    to_flush.append((inst, cur))
            for inst, cur in to_flush:
                self.sink.add(self._bucket(inst), cur)
                # flush 后清空这根，等待后续新 bar
                del self._cur[inst]
                self._last_recv_ms.pop(inst, None)

    def flush_all(self) -> None:
        with self._lock:
            for inst, rec in self._cur.items():
                self.sink.add(self._bucket(inst), rec)
            self._cur.clear()
            self._last_recv_ms.clear()

# ------------------------------ Worker ------------------------------ #

class OkxWsWorker(threading.Thread):
    def __init__(
        self,
        name: str,
        ws_url: str,
        sink: FileSink,
        subscribe_builder: Callable[[], List[Dict[str, Any]]],
        on_data: Callable[[Dict[str, Any], FileSink], None],
        log_level: str = "INFO",
        ping_interval: int = 20,
        ping_timeout: int = 10,
    ) -> None:
        super().__init__(daemon=True)
        self.name = name
        self.ws_url = ws_url
        self.sink = sink
        self.subscribe_builder = subscribe_builder
        self.on_data = on_data
        self.log_level = log_level.upper()
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self._stop = threading.Event()
        self._app: Optional[WebSocketApp] = None
        self._backoff = 1.0

    def _log(self, lvl: str, msg: str) -> None:
        order = ["DEBUG", "INFO", "WARN", "ERROR"]
        if order.index(lvl) >= order.index(self.log_level):
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}][{lvl}][{self.name}] {msg}")

    def _on_open(self, ws: WebSocketApp) -> None:
        self._log("INFO", f"[OPEN] -> {self.ws_url}")
        args = self.subscribe_builder()
        for i, arg in enumerate(args, 1):
            try:
                self._log("DEBUG", f"[SUB {i}/{len(args)}] {arg}")
                ws.send(json.dumps({"op": "subscribe", "args": [arg]}))
                time.sleep(0.02)
            except Exception as e:
                self._log("ERROR", f"Subscribe send failed: {e}")
        self._backoff = 1.0

    def _on_message(self, ws: WebSocketApp, message: str) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            return
        # 事件类
        if "event" in payload:
            # 更明确的错误提示
            if payload.get("event") == "error":
                self._log("ERROR", f"EventError: {payload}")
            else:
                self._log("INFO", f"Event: {payload}")
            return
        # 业务数据
        try:
            self.on_data(payload, self.sink)
        except Exception as e:
            self._log("WARN", f"on_data error: {e}; payload_head={str(payload)[:256]}")

    def _on_error(self, ws: WebSocketApp, error: Any) -> None:
        self._log("ERROR", f"WebSocket error: {error}")

    def _on_close(self, ws: WebSocketApp, status_code: Any, msg: Any) -> None:
        self._log("WARN", f"WebSocket closed: code={status_code}, msg={msg}")

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self._app = WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # ping 间隔略短一点 + timeout
                self._app.run_forever(ping_interval=self.ping_interval, ping_timeout=self.ping_timeout)
            except Exception as e:
                self._log("ERROR", f"run_forever exception: {e}")

            if self._stop.is_set():
                break

            # 带抖动的指数回退（最大 30s）
            backoff = min(self._backoff, 30.0)
            jitter = random.uniform(0, 0.3 * backoff)
            sleep_s = backoff + jitter
            self._log("INFO", f"Reconnecting in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
            self._backoff = min(self._backoff * 2.0, 30.0)

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._app:
                self._app.close()
        except Exception:
            pass

# --------------------------- Handlers --------------------------- #

def handle_depth_payload(payload: Dict[str, Any], sink: FileSink, depth_channel: str) -> None:
    arg = payload.get("arg", {})
    data = payload.get("data")
    if not arg or not data:
        return
    ch = arg.get("channel")
    inst = arg.get("instId")
    if not ch or not ch.startswith("books"):
        return
    for ob in data:
        ts_ms = int(ob.get("ts") or ob.get("t", utc_ms_now()))
        action = ob.get("action")
        checksum = ob.get("checksum")
        ts_iso = ts_ms_to_iso(ts_ms)
        for side_name in ("bids", "asks"):
            levels = ob.get(side_name, []) or []
            for lvl_idx, lvl in enumerate(levels):
                try:
                    px = float(lvl[0]); sz = float(lvl[1])
                except Exception:
                    continue
                rec = {
                    "ts": ts_ms,
                    "ts_iso": ts_iso,
                    "instId": inst,
                    "side": "bid" if side_name == "bids" else "ask",
                    "level": lvl_idx + 1,
                    "px": px,
                    "sz": sz,
                }
                if checksum is not None:
                    rec["checksum"] = checksum
                if action is not None:
                    rec["action"] = action
                bucket = f"depth/{depth_channel}/{inst}"
                sink.add(bucket, rec)

def make_candle_handler(candle_channel: str,
                        candle_mode: str,
                        sink: FileSink,
                        finalize_grace_mult: float) -> Callable[[Dict[str, Any], FileSink], None]:
    """
    - all : 每条推送都写
    - final: 仅最终bar（含超时定稿）
    """
    period = candle_period_ms(candle_channel)

    if candle_mode == "all":
        def handle_all(payload: Dict[str, Any], _sink: FileSink) -> None:
            arg = payload.get("arg", {})
            data = payload.get("data")
            if not arg or not data:
                return
            ch = arg.get("channel")
            inst = arg.get("instId")
            if ch != candle_channel:
                return
            for row in data:
                ts_ms = int(row[0])
                o, h, l, c = map(float, row[1:5])
                vol = float(row[5]) if len(row) > 5 and row[5] != "" else None
                rec = {"ts": ts_ms, "ts_iso": ts_ms_to_iso(ts_ms),
                       "instId": inst, "o": o, "h": h, "l": l, "c": c, "vol": vol}
                bucket = f"candles/{candle_channel}/{inst}"
                _sink.add(bucket, rec)
        return handle_all

    # final
    aggregator = CandleAggregator(sink, candle_channel, period_ms=period, grace_mult=finalize_grace_mult)

    def handle_final(payload: Dict[str, Any], _sink: FileSink) -> None:
        arg = payload.get("arg", {})
        data = payload.get("data")
        if not arg or not data:
            return
        ch = arg.get("channel")
        inst = arg.get("instId")
        if ch != candle_channel:
            return
        for row in data:
            ts_ms = int(row[0])
            o, h, l, c = map(float, row[1:5])
            vol = float(row[5]) if len(row) > 5 and row[5] != "" else None
            rec = {"ts": ts_ms, "ts_iso": ts_ms_to_iso(ts_ms),
                   "instId": inst, "o": o, "h": h, "l": l, "c": c, "vol": vol}
            aggregator.ingest(rec)

    # 暴露 flush/tick 给主循环
    handle_final.flush_aggregator = aggregator.flush_all          # type: ignore[attr-defined]
    handle_final.tick_aggregator  = aggregator.tick               # type: ignore[attr-defined]
    return handle_final

def handle_trades_payload(payload: Dict[str, Any], sink: FileSink) -> None:
    arg = payload.get("arg", {})
    data = payload.get("data")
    if not arg or not data:
        return
    if arg.get("channel") != "trades":
        return
    for t in data:
        try:
            ts_ms = int(t.get("ts", utc_ms_now()))
            rec = {
                "ts": ts_ms,
                "ts_iso": ts_ms_to_iso(ts_ms),
                "instId": t.get("instId"),
                "tradeId": t.get("tradeId"),
                "px": float(t.get("px")) if t.get("px") not in (None, "") else None,
                "sz": float(t.get("sz")) if t.get("sz") not in (None, "") else None,
                "side": t.get("side"),
            }
            bucket = f"trades/{t.get('instId')}"
            sink.add(bucket, rec)
        except Exception as e:
            print(f"[TRADES] parse error: {e}; row={t}", file=sys.stderr)

# --------------------------- Copy & Read --------------------------- #

def copy_range(base_subdir: str, src_root: str, dst_root: str, start: str, end: str) -> List[str]:
    copied = []
    for dt in iter_dates(start, end):
        y, m, d = ymd_parts_from_date(dt)
        rel = os.path.join(base_subdir, f"year={y}", f"month={m}", f"day={d}")
        src = os.path.join(src_root, rel)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(dst_root, rel)
        ensure_dir(dst)
        for name in os.listdir(src):
            s = os.path.join(src, name)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(dst, name))
        copied.append(dst)
    return copied

def read_range_df(base_subdir: str, root: str, start: str, end: str,
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
    if ds is None:
        raise RuntimeError("pyarrow is required to read partitioned dataset. pip install pyarrow")
    dataset_path = os.path.join(root, base_subdir)
    dataset = ds.dataset(dataset_path, format="parquet", partitioning="hive")

    Y = ds.field("year").cast("string")
    M = ds.field("month").cast("string")
    D = ds.field("day").cast("string")

    preds = []
    for dt in iter_dates(start, end):
        preds.append((Y == str(dt.year)) & (M == f"{dt.month:02d}") & (D == f"{dt.day:02d}"))
    if not preds:
        return pd.DataFrame()
    flt = preds[0]
    for p in preds[1:]:
        flt = flt | p

    table = dataset.to_table(filter=flt, columns=columns)
    return table.to_pandas()

# ------------------------------ CLI ------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OKX streamer + partitioned dataset utilities (stable)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("stream", help="Start streaming WS data to storage")
    ps.add_argument("--inst", nargs="+", required=True,
                    help="Instrument IDs, e.g. BTC-USDT-SWAP ETH-USDT BTC-USDT")
    ps.add_argument("--depth-channel", default="books5",
                    choices=["books", "books5", "books50-l2-tbt", "books-l2-tbt"],
                    help="Depth channel (public WS)")
    ps.add_argument("--depth-freq", choices=["100ms", "1s"],
                    help="Depth update frequency")
    ps.add_argument("--candle", default="1m",
                    help="Candle: 1m/3m/5m/15m/30m/1H/4H/1D")
    ps.add_argument("--candle-mode", default="final", choices=["final", "all"],
                    help="How to save candles: only final bar or all updates")
    ps.add_argument("--finalize-grace-mult", type=float, default=2.0,
                    help="Final mode timeout multiplier of candle period (default 2.0)")
    ps.add_argument("--no-candles", action="store_true", help="Disable candle subscription")
    ps.add_argument("--ticks", action="store_true", help="Enable trades (tick) subscription")
    ps.add_argument("--format", default="feather", choices=["feather", "parquet"],
                    help="Output file format for NORMAL mode (non-dataset)")
    ps.add_argument("--dataset", choices=["parquet"],
                    help="Enable partitioned dataset append mode (parquet only)")
    ps.add_argument("--out", default="./okx_data", help="Output directory root")
    ps.add_argument("--chunk-rows", type=int, default=2000, help="Rows per buffer flush")
    ps.add_argument("--flush-secs", type=int, default=10, help="Flush interval seconds")
    ps.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARN", "ERROR"], help="Log verbosity")
    ps.add_argument("--ws-public", default=DEFAULT_WS_PUBLIC, help="Public WS URL (depth/trades)")
    ps.add_argument("--ws-business", default=DEFAULT_WS_BUSINESS, help="Business WS URL (candles)")

    pc = sub.add_parser("copy-range", help="Copy a date range of partitions to another root")
    pc.add_argument("--base-subdir", required=True, help="e.g. candles/candle1m/BTC-USDT")
    pc.add_argument("--src", required=True, help="Source dataset root")
    pc.add_argument("--dst", required=True, help="Destination dataset root")
    pc.add_argument("--start", required=True, help="Start date YYYY-MM-DD (UTC)")
    pc.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC)")

    pr = sub.add_parser("read-range", help="Read a date range from partitioned dataset")
    pr.add_argument("--base-subdir", required=True, help="e.g. depth/books5/BTC-USDT-SWAP")
    pr.add_argument("--root", required=True, help="Dataset root directory")
    pr.add_argument("--start", required=True, help="Start date YYYY-MM-DD (UTC)")
    pr.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC)")
    pr.add_argument("--columns", nargs="*", help="Optional projected columns")

    return p

# ------------------------------ Stream ------------------------------ #

def run_stream(args: argparse.Namespace) -> None:
    sink = FileSink(
        out_root=args.out,
        file_format=args.format,
        dataset=args.dataset,
        chunk_rows=args.chunk_rows,
        flush_secs=args.flush_secs,
    )

    workers: List[OkxWsWorker] = []

    # Depth
    def depth_subs() -> List[Dict[str, Any]]:
        if args.depth_freq:
            return [{"channel": args.depth_channel, "instId": inst, "freq": args.depth_freq}
                    for inst in args.inst]
        else:
            return [{"channel": args.depth_channel, "instId": inst} for inst in args.inst]

    depth_worker = OkxWsWorker(
        name="DEPTH",
        ws_url=args.ws_public,
        sink=sink,
        subscribe_builder=depth_subs,
        on_data=lambda payload, s: handle_depth_payload(payload, s, args.depth_channel),
        log_level=args.log_level,
    )
    workers.append(depth_worker)

    # Candles
    candle_handler_flush = None
    candle_handler_tick = None
    if not args.no_candles:
        candle_channel = normalize_candle_choice(args.candle)
        def candle_subs() -> List[Dict[str, Any]]:
            return [{"channel": candle_channel, "instId": inst} for inst in args.inst]

        candle_handler = make_candle_handler(
            candle_channel, args.candle_mode, sink, finalize_grace_mult=args.finalize_grace_mult
        )
        candle_handler_flush = getattr(candle_handler, "flush_aggregator", None)
        candle_handler_tick  = getattr(candle_handler, "tick_aggregator", None)

        candle_worker = OkxWsWorker(
            name=f"CANDLE[{candle_channel}]",
            ws_url=args.ws_business,
            sink=sink,
            subscribe_builder=candle_subs,
            on_data=candle_handler,
            log_level=args.log_level,
        )
        workers.append(candle_worker)

    # Trades
    if args.ticks:
        def trades_subs() -> List[Dict[str, Any]]:
            return [{"channel": "trades", "instId": inst} for inst in args.inst]
        trades_worker = OkxWsWorker(
            name="TRADES",
            ws_url=args.ws_public,
            sink=sink,
            subscribe_builder=trades_subs,
            on_data=handle_trades_payload,
            log_level=args.log_level,
        )
        workers.append(trades_worker)

    # Graceful shutdown
    stop_event = threading.Event()
    def handle_sig(_signo, _frame):
        print("[*] SIG received, stopping...")
        stop_event.set()
        for w in workers:
            w.stop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_sig)

    # Start
    for w in workers:
        w.start()

    # Main loop with candle timeout tick
    try:
        while not stop_event.is_set():
            if callable(candle_handler_tick):
                candle_handler_tick()
            time.sleep(0.5)
    finally:
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=5)
        if callable(candle_handler_flush):
            candle_handler_flush()
        sink.flush_all()
        print("[*] Stopped and flushed.")

# ------------------------------ Entry ------------------------------ #

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "stream":
        run_stream(args)
    elif args.cmd == "copy-range":
        copied = copy_range(args.base_subdir, args.src, args.dst, args.start, args.end)
        print("Copied partitions:")
        for pth in copied:
            print("  ", pth)
    elif args.cmd == "read-range":
        df = read_range_df(args.base_subdir, args.root, args.start, args.end, args.columns)
        print(df.head())
        print(f"[INFO] Loaded rows: {len(df)}")
    else:
        parser.error("Unknown command")

if __name__ == "__main__":
    main()
