import os
import sys
import importlib
import pandas as pd

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__))))

from datetime import datetime, timezone, timedelta
from typing import Union

from coin_python.config import DataConfig


def to_milliseconds(time: datetime):
    return int(time.timestamp()) * 1000


def get_previous_day_timestamps_utc(date_input=None):
    """
    获取给定日期（或今天）的前一天 UTC0 零点时间戳 和 23:59:59 时间戳。

    :param date_input: None（默认今天）、datetime 对象、时间戳（int）、或字符串（"YYYY-MM-DD"）
    :return: (零点时间戳, 23:59:59 时间戳)
    """
    # 处理输入类型
    if date_input is None:
        base_dt = datetime.now(timezone.utc)
    elif isinstance(date_input, int):  # 时间戳
        base_dt = datetime.fromtimestamp(date_input, tz=timezone.utc)
    elif isinstance(date_input, str):  # 日期字符串
        base_dt = datetime.strptime(date_input, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif isinstance(date_input, datetime):  # datetime
        base_dt = date_input.astimezone(timezone.utc)
    else:
        raise ValueError("Unsupported date_input type")

    # 获取当天 UTC 零点
    today_midnight = datetime(base_dt.year, base_dt.month, base_dt.day, tzinfo=timezone.utc)

    # 前两天23：59：59
    prev_midnight = today_midnight - timedelta(days=1, seconds=1)
    # 前一天 23:59:59
    prev_end_of_day = today_midnight - timedelta(seconds=1)

    return int(prev_midnight.timestamp()) * 1000, int(prev_end_of_day.timestamp()) * 1000


def get_today_zero_to_newest_timestamp():
    now_utc = datetime.now(timezone.utc)
    today_zero_utc = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
    prev_end_of_day = today_zero_utc - timedelta(seconds=1)
    return int(prev_end_of_day.timestamp()) * 1000, int(now_utc.timestamp()) * 1000


def update_historical_k_line(exchange: str, mea: str, key_path: str, symbol: str, after: str, before: str,
                             save_path: str, bar: str = '1m'):
    exchange_model = getattr(importlib.import_module(f"coin_python.exchange_module.{exchange}"), "PrivateClient")
    if not os.path.exists(os.path.join(save_path, symbol)):
        os.mkdir(os.path.join(save_path, symbol))
    model = exchange_model(key_path, mea)
    data = model.get_his_k_line(bar, symbol, after, before)
    data.to_csv(os.path.join(save_path, symbol,
                             f'{datetime.fromtimestamp(after / 1000, tz=timezone.utc).strftime("%Y%m%d")}.csv'),
                index=False)
    print(
        f'{symbol} updating {datetime.fromtimestamp(after / 1000, tz=timezone.utc).strftime("%Y%m%d")} data finished...')


def get_historical_k_line_local(symbol: str, start_date: Union[str, datetime.date] = None,
                                end_date: Union[str, datetime.date] = None, pre_days: int = 20):
    if start_date is None or end_date is None:
        end_date = (datetime.today() - timedelta(days=1)).date().strftime('%Y%m%d')
        start_date = (datetime.today() - timedelta(days=pre_days)).date().strftime('%Y%m%d')
    else:
        end_date = pd.to_datetime(end_date).strftime('%Y%m%d')
        start_date = pd.to_datetime(start_date).strftime('%Y%m%d')
    # your own saved path
    read_files_path = os.path.join(DataConfig.OKX_HIS_DATA_BASE_PATH, symbol)
    return pd.concat([pd.read_csv(os.path.join(read_files_path, f"{date_.strftime('%Y%m%d')}.csv")) for date_ in
                      pd.period_range(start_date, end_date)])


def check_1min_files(saved_path: str):
    symbols_list = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'XRP-USDT-SWAP', 'USDC-USDT-SWAP',
                    'BNB-USDT-SWAP'] + ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'XRP-USDT', 'USDC-USDT', 'BNB-USDT']
    for symbol in symbols_list:
        whole_files = [os.path.join(f'{saved_path}/{symbol}', file) for
                       root, dirs, files in os.walk(f'{saved_path}/{symbol}') for
                       file in files]
        for file in sorted(whole_files):
            if pd.read_feather(file).shape[0] < 1440:
                print(file)
        print('-' * 100)
        print(f'{symbol} end ...')


if __name__ == '__main__':
    check_1min_files()
