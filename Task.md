# Data Preprocessing & Feature Engineering Pipeline Specification (v2.0)

> **Version**: 2.0  
> **Last Updated**: 2026-03-23  
> **Author**: YC  
> **Objective**: Generate slippage research dataset by backtracking 60s of market microstructure data from Trade Records

---

## 1. Project Overview

This project aims to build a high-performance data preprocessing pipeline with the following core objectives:

1. **Data Alignment**: Use `Trade Records` as anchor points to retrospectively extract 60-second windows of Orderbook snapshots and Trade Flow data
2. **Feature Engineering**: Extract liquidity, volatility, and microstructure features
3. **Label Construction**: Use `gain_vs_threshold` as primary label, supplemented with execution delay, basis spread, and other derived metrics
4. **TickSize Integration**: Incorporate Binance contract/spot minimum price increments to generate standardized features

**Output**: CSV files grouped by `trade_mode` (0/2) and `symbol` for supervised learning model training.

---

## 2. Data Source Specifications

### 2.1 Trade Records (Label Source)

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `date` | datetime | Record timestamp | ✅ |
| `symbol` | str | Trading pair, e.g., `1MBABYDOGE` | ✅ |
| `trade_mode` | int | 0=Maker-Spot/Taker-Swap, 2=Other modes | ✅ (filter: ∈{0,2}) |
| `gain_vs_threshold` | float | Primary Label: actual gain - threshold | ✅ |
| `threshold` | float | Cost threshold = ln(swap_price) - ln(spot_price) | ✅ (new) |
| `anticipated_basis` | float | Expected basis spread before execution | ✅ (new) |
| `executed_basis` | float | Actual executed basis spread | ✅ (new) |
| `timer_start_ts` | int64 | Strategy trigger timestamp (ms) | ✅ (new) |
| `maker/spot_executed_ts` | int64 | Maker-side execution timestamp (ms) | ✅ (new) |
| `taker/swap/haircut_executed_ts` | int64 | Taker-side execution timestamp (ms) | ✅ (new) |

> ⚠️ **Note**: Fields containing `/` follow original data format; code should normalize to underscore naming (e.g., `maker_spot_executed_ts`)

### 2.2 Orderbook & Trade Flow (Feature Sources)

| Data Type | Path Template | Time Field | Key Fields |
|-----------|--------------|------------|------------|
| **Orderbook** | `.../market_processed/{date}/{symbol}USDT/book_{symbol}USDT_{date}.csv.gz` | `time_str` (ISO8601) | `spot_bid1_px`, `spot_ask1_px`, `swap_bid1_px`, `swap_ask1_px`, `funding_rate`, `index_price` |
| **Trade Flow** | `.../market_processed/{date}/{symbol}USDT/trade_{symbol}USDT_{date}.csv.gz` | `time_str` (ISO8601) | `bid_px`, `ask_px`, `trade_type`, `local_ts` |

### 2.3 Binance API (TickSize Metadata)

| Endpoint | Purpose | Key Response Fields |
|----------|---------|-------------------|
| **Spot**: `https://api.binance.com/api/v3/exchangeInfo` | Fetch spot minimum price increment | `symbols[].filters[?(@.filterType=='PRICE_FILTER')].tickSize` |
| **Swap**: `https://fapi.binance.com/fapi/v1/exchangeInfo` | Fetch futures minimum price increment | `symbols[].filters[?(@.filterType=='PRICE_FILTER')].tickSize` |

---

## 3. Processing Logic & Pipeline

### 3.1 Overall Workflow

```mermaid
graph TD
    A[Load Trade Records] --> B[Filter trade_mode ∈ {0,2}]
    B --> C[Parse timestamp fields]
    C --> D[Calculate execute_delay]
    D --> E[Group by symbol+date]
    E --> F[Load corresponding Orderbook/Trade Flow]
    F --> G[60s window backtracking & alignment]
    G --> H[Generate base features]
    H --> I[Integrate TickSize-standardized features]
    I --> J[Merge Labels + Features]
    J --> K[Save CSV by mode/symbol]
```

### 3.2 Time Alignment Logic

```python
# Pseudocode: Window extraction
def extract_window(trade_record, orderbook_df, trade_flow_df, window_sec=60):
    # Reference time: taker execution timestamp
    t_exec = trade_record['taker_swap_haircut_executed_ts']  # ms
    t_start = t_exec - window_sec * 1000
    
    # Convert time_str → ms timestamp
    orderbook_df['ts_ms'] = pd.to_datetime(orderbook_df['time_str']).astype(int) // 10**6
    trade_flow_df['ts_ms'] = pd.to_datetime(trade_flow_df['time_str']).astype(int) // 10**6
    
    # Filter window data
    ob_window = orderbook_df[(orderbook_df['ts_ms'] >= t_start) & (orderbook_df['ts_ms'] <= t_exec)]
    tf_window = trade_flow_df[(trade_flow_df['ts_ms'] >= t_start) & (trade_flow_df['ts_ms'] <= t_exec)]
    
    return ob_window, tf_window
```

### 3.3 New Field Calculations

| Field | Calculation Logic | Unit | Description |
|-------|------------------|------|-------------|
| `execute_delay` | `taker_swap_haircut_executed_ts - timer_start_ts` | ms | Strategy execution latency; measures system response speed |
| `threshold` | `ln(swap_price) - ln(spot_price)` | - | Theoretical arbitrage threshold (log spread) |
| `basis_expected` | `anticipated_basis` (original field) | - | Basis spread observed before execution |
| `basis_executed` | `executed_basis` (original field) | - | Actual basis spread at execution |
| `basis_slippage` | `basis_executed - basis_expected` | - | Basis slippage (core research target) |

---

## 4. TickSize Integration Scheme

### 4.1 TickSize Fetching & Caching

```python
# util/binance_meta.py
import requests
from functools import lru_cache

@lru_cache(maxsize=128)
def get_ticksize(symbol: str, market_type: str) -> float:
    """
    Fetch minimum price increment for specified trading pair
    market_type: 'spot' or 'swap'
    """
    url = {
        'spot': 'https://api.binance.com/api/v3/exchangeInfo',
        'swap': 'https://fapi.binance.com/fapi/v1/exchangeInfo'
    }[market_type]
    
    resp = requests.get(url, params={'symbol': symbol + 'USDT'}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    for s in data['symbols']:
        if s['symbol'] == symbol + 'USDT':
            for f in s['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    return float(f['tickSize'])
    raise ValueError(f"TickSize not found for {symbol}")
```

### 4.2 TickSize-Standardized Features (New)

| Feature Category | Feature Name | Calculation Logic | Significance |
|-----------------|--------------|------------------|--------------|
| **Price Normalization** | `spot_price_ticks` | `spot_bid1_px / spot_ticksize` | Price as multiple of minimum increment |
| **Spread Normalization** | `spread_ticks` | `(ask1_px - bid1_px) / ticksize` | Bid-ask spread in tick units |
| **Slippage Normalization** | `slippage_ticks` | `abs(basis_executed - basis_expected) / avg_ticksize` | Slippage in minimum increment units |
| **Depth Normalization** | `depth1_ticks` | `bid1_qty * bid1_px / ticksize` | Top-of-book depth normalized by price unit |
| **Volatility Normalization** | `volatility_ticks` | `std(midprice_60s) / ticksize` | 60s price volatility in tick units |

> 💡 **Advantage**: Assets across different price ranges (e.g., 0.0005 vs 50.0) become horizontally comparable after tick normalization

### 4.3 Feature Generation Function Example

```python
# data/feature_engineering.py
def generate_features(ob_window, tf_window, trade_record, spot_tick, swap_tick):
    features = {}
    
    # === Base statistical features ===
    midprice = (ob_window['spot_bid1_px'] + ob_window['spot_ask1_px']) / 2
    features['midprice_mean'] = midprice.mean()
    features['midprice_std'] = midprice.std()
    
    # === TickSize-normalized features ===
    spread = ob_window['spot_ask1_px'] - ob_window['spot_bid1_px']
    features['spread_ticks'] = (spread / spot_tick).mean()
    
    # === Execution-related features ===
    features['execute_delay_ms'] = trade_record['taker_swap_haircut_executed_ts'] - \
                                   trade_record['timer_start_ts']
    features['basis_slippage'] = trade_record['executed_basis'] - \
                                 trade_record['anticipated_basis']
    features['basis_slippage_ticks'] = features['basis_slippage'] / ((spot_tick + swap_tick)/2)
    
    # === Liquidity features ===
    depth_imbalance = (ob_window['spot_bid1_qty'] - ob_window['spot_ask1_qty']) / \
                      (ob_window['spot_bid1_qty'] + ob_window['spot_ask1_qty'] + 1e-8)
    features['depth_imbalance_mean'] = depth_imbalance.mean()
    
    return pd.Series(features)
```

---

## 5. Output Data Specifications

### 5.1 Save Paths

```
quant_toolbox/
└── dataset/
    └── preprocessed/
        ├── mode0/
        │   ├── sample_1MBABYDOGE.csv
        │   ├── sample_ADA.csv
        │   └── ...
        └── mode2/
            ├── sample_1MBABYDOGE.csv
            └── ...
```

### 5.2 Output Field List (CSV Columns)

| Category | Field Name | Type | Description |
|----------|------------|------|-------------|
| **Label** | `gain_vs_threshold` | float | Primary target variable |
| **Label** | `basis_slippage` | float | Basis spread slippage (derived label) |
| **Meta** | `symbol` | str | Trading pair |
| **Meta** | `trade_mode` | int | 0 or 2 |
| **Meta** | `exec_ts_utc` | datetime | Execution time (normalized) |
| **Time** | `execute_delay_ms` | int | Execution latency |
| **Time** | `timer_start_ts` | int | Strategy trigger timestamp |
| **Price** | `threshold` | float | Theoretical threshold |
| **Price** | `basis_expected` | float | Expected basis spread |
| **Price** | `basis_executed` | float | Actual basis spread |
| **Feature** | `spot_bid1_px`, `spot_ask1_px` | float | Orderbook at execution moment |
| **Feature** | `spread_ticks` | float | Normalized spread |
| **Feature** | `midprice_volatility_ticks` | float | Normalized volatility |
| **Feature** | `depth_imbalance_mean` | float | Depth imbalance metric |
| **Feature** | `trade_volume_60s` | float | 60s trading volume |
| **Feature** | `orderbook_slope` | float | Orderbook slope indicator |
| **...** | *(extensible)* | | |

---

## 6. Project Structure Updates

```text
quant_toolbox/
├── data/
│   ├── binance_downloader.py
│   ├── preprocess.py           # Main entry: coordinate data loading & saving
│   ├── feature_engineering.py  # [Core] Feature computation logic
│   └── binance_meta.py         # [New] TickSize fetching & caching
├── util/
│   ├── logger.py
│   ├── config.py               # Add: BINANCE_SPOT_URL, BINANCE_SWAP_URL
│   └── helpers.py              # Add: parse_timestamp(), normalize_field_names()
├── script/
│   ├── run_factor_gen.py       # [New] CLI entry point
│   └── test_ticksize.py        # [New] TickSize fetch test
├── dataset/
│   └── preprocessed/           # [New] Output directory
│       ├── mode0/
│       └── mode2/
├── config/
│   └── .env                    # Store API keys (if auth required)
├── requirements.txt            # Add: requests, polars (optional)
└── README.md                   # Update usage documentation
```

---

## 7. Technical Implementation Details

### 7.1 Performance Optimization Strategies

| Problem | Solution |
|---------|----------|
| **Slow large file reads** | Use `polars.read_csv()` instead of pandas, or chunk by date |
| **Inefficient time alignment** | Pre-sort + `merge_asof` (pandas) or `join_asof` (polars) |
| **Repeated TickSize requests** | `@lru_cache` + local JSON cache file (24h expiry) |
| **Memory overflow** | Stream process by `symbol`, `del` + `gc.collect()` after each batch |

### 7.2 Field Name Normalization Mapping

```python
# util/helpers.py
FIELD_MAPPING = {
    'maker/spot_executed_ts': 'maker_spot_executed_ts',
    'taker/swap/haircut_executed_ts': 'taker_swap_haircut_executed_ts',
    'anticipated_basis.1': 'anticipated_basis_secondary',  # Handle duplicate columns
    # ... other mappings
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=FIELD_MAPPING)
```

### 7.3 Error Handling

- **TickSize fetch failure**: Fallback to default value (e.g., `1e-8`), log warning
- **Empty time window**: Fill with `NaN` and flag `window_missing=True` for later filtering
- **Missing fields**: Validate completeness at start of `preprocess.py`, fail early with clear error

---

## 8. Testing & Validation Plan

### 8.1 Unit Tests (pytest)

```python
# tests/test_feature_engineering.py
def test_execute_delay_calculation():
    record = {'timer_start_ts': 1000, 'taker_swap_haircut_executed_ts': 1500}
    assert calculate_delay(record) == 500

def test_ticksize_standardization():
    spread = 0.0001
    tick = 1e-5
    assert spread_to_ticks(spread, tick) == 10.0
```

### 8.2 Integration Test (Using Uploaded Samples)

```bash
# Validate full pipeline with sample data
python script/run_factor_gen.py \
  --trade-file dataset/bn_trade/combined_example.csv.csv \
  --market-root kronos_test/Kronos/dataset/market_processed \
  --output-dir dataset/preprocessed \
  --symbol 1MBABYDOGE \
  --date 20251227 \
  --dry-run  # Print logs only, no save
```

### 8.3 Output Validation Checklist

- [ ] CSV files generated at `mode{0,2}/sample_{symbol}.csv` paths
- [ ] Each record includes `gain_vs_threshold` + 6 new fields
- [ ] `execute_delay` calculated correctly (unit: ms)
- [ ] TickSize features within reasonable range (e.g., `spread_ticks > 0`)
- [ ] Timestamp alignment error < 100ms

---

## 9. Next Steps & Timeline

| Phase | Task | Estimated Time | Deliverable |
|-------|------|---------------|-------------|
| **Phase 1** | Implement `binance_meta.py` + TickSize caching | 2h | Reusable ticksize fetch function |
| **Phase 2** | Develop `feature_engineering.py` core logic | 4h | Single-record backtrack + feature function |
| **Phase 3** | Integrate into `preprocess.py` + batch processing | 3h | Multi-symbol pipeline support |
| **Phase 4** | Write tests + documentation + example scripts | 2h | Runnable `run_factor_gen.py` |
| **Phase 5** | End-to-end validation with sample data | 1h | Validation report + sample output CSV |

**Total**: ~12 hours development + testing

---

## 10. Appendix: Binance API Call Examples

### 10.1 Spot TickSize

```bash
curl -s "https://api.binance.com/api/v3/exchangeInfo?symbol=1MBABYDOGEUSDT" | jq '.symbols[0].filters[] | select(.filterType=="PRICE_FILTER") | .tickSize'
# Output: "0.0000001"
```

### 10.2 Swap TickSize

```bash
curl -s "https://fapi.binance.com/fapi/v1/exchangeInfo?symbol=1MBABYDOGEUSDT" | jq '.symbols[0].filters[] | select(.filterType=="PRICE_FILTER") | .tickSize'
# Output: "0.0000001"
```

> 💡 **Recommendation**: Pre-fetch ticksize for all target symbols on first run and save to `config/ticksize_cache.json` to avoid repeated requests hitting rate limits

---

> ✅ **Please review the above specification**.  
> If confirmed, I will proceed with code implementation per this document; if adjustments are needed (e.g., feature definitions, field naming, output format), please let me know.