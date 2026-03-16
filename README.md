# quant_toolbox

> A Python toolkit for **Cryptocurrency Order Book Analysis & Modeling** —
> download live Binance order-book data, engineer micro-structure features,
> and plug in your own prediction models.

---

## Directory Structure

```
quant_toolbox/
├── data/
│   ├── binance_downloader.py   # Download order-book data (REST / WebSocket)
│   └── preprocess.py           # Cleaning, feature engineering, Parquet/CSV export
├── util/
│   ├── logger.py               # Unified logging configuration
│   ├── config.py               # Settings via pydantic-settings + .env
│   └── helpers.py              # Time utilities & file I/O helpers
├── script/
│   └── temp_test.py            # Hello World / environment sanity check
├── src/
│   ├── __init__.py
│   ├── trainer.py              # Model training pipeline skeleton
│   └── inference.py            # Model inference pipeline skeleton
├── model/
│   └── .gitkeep                # Preserved empty directory (weights are git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Yechang618/quant_toolbox.git
cd quant_toolbox

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Variables

**Never hard-code API keys.** Create a `.env` file in the project root (it is
git-ignored):

```dotenv
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Optional overrides
LOG_LEVEL=INFO
DATA_RAW_DIR=data/raw
DATA_PROCESSED_DIR=data/processed
MODEL_DIR=model
```

---

## Usage

### Sanity check

```bash
python script/temp_test.py
```

### Download order-book snapshots (REST)

```bash
python data/binance_downloader.py \
    --symbol BTC/USDT \
    --mode rest \
    --num-snapshots 200 \
    --output-dir data/raw
```

### Stream order-book updates (WebSocket)

```bash
python data/binance_downloader.py \
    --symbol ETH/USDT \
    --mode ws \
    --duration 120 \
    --output-dir data/raw
```

### Preprocess raw data

```bash
python data/preprocess.py data/raw/BTC_USDT_orderbook.jsonl \
    --output-dir data/processed \
    --fmt parquet
```

### Train a model

```bash
python src/trainer.py \
    --data-path data/processed/BTC_USDT_orderbook_processed.parquet \
    --model-out model/btc_model.pth
```

### Run inference

```bash
python src/inference.py \
    --model-path model/btc_model.pth \
    --data-path data/processed/BTC_USDT_orderbook_processed.parquet \
    --output-path data/processed/predictions.csv
```

---

## Security Notes

* **API keys must never be committed** to the repository.  Always load them
  from environment variables or a `.env` file.
* The `model/` directory and all contents of `data/raw/` and
  `data/processed/` are listed in `.gitignore` to prevent accidental upload
  of sensitive data or large binary files.

---

## License

See [LICENSE](LICENSE).
