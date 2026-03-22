import hashlib
import hmac
import requests
import time
import urllib.parse
import os
import base64
import pandas as pd

from datetime import datetime, timedelta
from copy import deepcopy

from coin_python.base.auth import KeyAuthor
from coin_python.base.base_tools import Query
from enum import Enum


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class InstType(Enum):
    SPOT = "SPOT"
    SWAP = "SWAP"
    FUTURES = "FUTURES"
    OPTIONS = "OPTIONS"


class TradeMode(Enum):
    CROSS = "cross"
    ISOLATED = "isolated"
    CASH = "cash"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    POST_ONLY = "post_only"
    FOK = "fok"
    IOC = "ioc"
    OPTIMAL_LIMIT_IOC = "optimal_limit_ioc"


class OkexAuth(requests.auth.AuthBase):
    def __init__(self, key, use_simulate=False):
        self._key = KeyAuthor.from_file(key)
        self._key.passphrase = self._key.get_value('passphrase')
        self._use_simulate = use_simulate

    def get_sign(self, str_to_be_signed):
        signature = hmac.new(self._key.secret_key_bytes,
                             str_to_be_signed.encode(),
                             digestmod=hashlib.sha256)
        signature = base64.b64encode(signature.digest()).decode()
        return signature

    def __call__(self, req: requests.PreparedRequest):
        assert req.method in ('GET', 'POST', 'DELETE'), req.method
        timestamp = '%.03f' % time.time()
        parsed_result = urllib.parse.urlparse(req.url)
        if req.method == 'GET' and parsed_result.query:
            path = parsed_result.path + '?' + parsed_result.query
        else:
            path = parsed_result.path

        body = req.body.decode() if req.body else ''
        str_to_be_signed = ''.join([timestamp, req.method, path, body])
        if not self._use_simulate:
            headers = {
                'OK-ACCESS-KEY': self._key.access_key,
                'OK-ACCESS-SIGN': self.get_sign(str_to_be_signed),
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': self._key.passphrase,
                'Content-Type': 'application/json',
            }
        else:
            headers = {
                'OK-ACCESS-KEY': self._key.access_key,
                'OK-ACCESS-SIGN': self.get_sign(str_to_be_signed),
                'OK-ACCESS-TIMESTAMP': timestamp,
                'OK-ACCESS-PASSPHRASE': self._key.passphrase,
                'Content-Type': 'application/json',
                'x-simulated-trading': 1
            }
        req.headers.update(headers)
        req.prepare_headers(req.headers)

        # print(req.headers)
        # print(req.body)
        # print(req.url)

        return req


class PrivateClient():
    def __init__(self, key_path, mea):
        self._key = KeyAuthor.from_file(key_path)
        self._auth = OkexAuth(key_path)
        self._mea = mea  # be all one
        self._url = "https://www.okx.com"
        self._query = Query(api_host=self._url, auth=self._auth)
        if 'Spot' in mea:
            self._inst_type = InstType.SPOT.value
        elif 'Futures' in mea:
            if 'swap' in mea:
                self._inst_type = InstType.SWAP.value
            else:
                self._inst_type = InstType.FUTURES.value
        elif 'Options' in mea:
            self._inst_type = InstType.FUTURES.value
        else:
            assert False, "unsupported mea"

    def to_native_symbol(self, base: str, quote: str):
        base = base.upper()
        quote = quote.upper()
        return "{}-{}".format(base, quote)

    def get_account_config(self):
        method = "GET"
        path = '/api/v5/account/config'
        params = {
        }
        response = self._query.query(
            method=method, path=path, params=params)
        return response.json()

    def get_balance(self, ccy: str = 'USDT'):
        method = "GET"
        path = '/api/v5/account/balance'
        params = {
            "ccy": ccy,
        }
        response = self._query.query(
            method=method, path=path, params=params)
        return response.json()

    def get_position(self, inst_type: str = None):
        method = "GET"
        path = '/api/v5/account/positions'
        params = {
            "instType": inst_type,
        }
        response = self._query.query(
            method=method, path=path, params=params)
        return response.json()

    def get_position_history(self,
                             inst_type: str = None,
                             inst_name: str = None,
                             after: int = None,
                             before: int = None):
        # Query api not include after and before
        method = "GET"
        path = '/api/v5/account/positions-history'
        params = {
            "instType": inst_type,
            "instId": inst_name,
            "after": after,
            "before": before,
        }
        response = self._query.query(
            method=method, path=path, params=params)
        return response.json()

    def query_transfer(self, begin_ms, end_ms):
        method = "GET"
        path = "/api/v5/asset/bills"
        params = {"before": begin_ms, "after": end_ms}
        response = self._query.query(
            method=method, path=path, params=params)
        return response.json()

    def query_7days_transfer(self):
        method = "GET"
        path = "/api/v5/account/bills"
        response = self._query.query(
            method=method, path=path)
        return response.json()

    def query_bills(self, begin_ms, end_ms):
        method = "GET"
        path = "/api/v5/asset/subaccount/bills"
        params = {"before": begin_ms, "after": end_ms}
        response = self._query.query(
            method=method, path=path, params=params)
        return response.json()["data"]

    def get_funding_fee(self, start_timestamp: int, end_timestamp: int, limit: int = 200):
        method = "GET"
        path = "/api/v5/account/bills-archive"

        if self._mea == "Futures.Okex.v5":
            inst_type = "FUTURES"
        elif self._mea == "Futures.Okex.v5-swap":
            inst_type = "SWAP"
        else:
            assert False, f"Unexpected mea:{self._mea}!"

        params = {
            "instType": inst_type,
            "type": 8,
            "begin": start_timestamp // 1000000,
            "end": end_timestamp // 1000000,
            "limit": limit
        }
        response = self._query.query(
            method=method, path=path, params=params)
        response.raise_for_status()
        return response.json()

    def query_deposit_history(self):
        method = "GET"
        path = "/api/v5/asset/deposit-history"
        response = self._query.query(
            method=method, path=path)
        return response.json()

    def query_withdrawal_history(self):
        method = "GET"
        path = "/api/v5/asset/withdrawal-history"
        response = self._query.query(
            method=method, path=path)
        return response.json()

    def query_withdrawal_history_with_timestamp(self, start_: int, end_: int):
        method = "GET"
        path = "/api/v5/asset/withdrawal-history"
        params = {
            'before': end_,
            'after': start_,
        }
        response = self._query.query(
            method=method, path=path, params=params)
        return response.json()

    def query_deposite_history_with_timestamp(self, start_: int, end_: int):
        method = "GET"
        path = "/api/v5/asset/deposit-history"
        params = {
            'before': end_,
            'after': start_,
        }
        response = self._query.query(
            method=method, path=path, params=params)
        return response.json()

    def get_open_order(self):
        method = "GET"
        path = '/api/v5/trade/orders-pending'
        params = {
            "instType": self._inst_type,
        }
        response = self._query.query(
            method=method, path=path, params=params)
        return response.json()

    def get_hist_order(self):
        method = "GET"
        path = '/api/v5/trade/orders-history'
        params = {
            "instType": self._inst_type,
        }
        response = self._query.query(
            method=method, path=path, params=params)
        return response.json()

    # begin_timestamp < end_timestamp
    def get_order_fill(self, symbol, begin_ms, end_ms):
        method = "GET"
        path = "/api/v5/trade/fills-history"
        params = {
            # "limit": 100,
            "instId": symbol,
            "instType": "SPOT",
            "begin": begin_ms,
            "end": end_ms
        }

        resp = self._query.query(method=method, path=path, params=params)

        outline_map = {
            "fill_id": "tradeId",
            "qty": "fillSz",
            "price": "fillPx",
            "fee_currency": "feeCcy",
            "fee_qty": "fee",
            "time_ms": "fillTime"
        }

        resp = resp.json()["data"]
        time.sleep(0.1)
        return resp, outline_map

    # it has 1 more args than above one
    def get_order_fill_with_inst(self, symbol, begin_ms, end_ms, instType):
        method = "GET"
        path = "/api/v5/trade/fills-history"
        params = {
            # "limit": 100,
            "instId": symbol,
            "instType": f"{instType}",
            "begin": begin_ms,
            "end": end_ms
        }

        resp = self._query.query(method=method, path=path, params=params)

        outline_map = {
            "fill_id": "tradeId",
            "qty": "fillSz",
            "price": "fillPx",
            "fee_currency": "feeCcy",
            "fee_qty": "fee",
            "time_ms": "fillTime"
        }

        resp = resp.json()["data"]
        time.sleep(0.1)
        return resp, outline_map

    def submit_order(
            self,
            inst_id: str,
            td_mode: TradeMode,
            side: Side,
            ord_type: OrderType,
            sz: str,
            px: str = None,
            posSide: str = None,
            client_ord_id: str = None,
    ):
        assert inst_id, "inst_id must be not None and not empty!"
        assert td_mode, "td_mode must be not None!"
        assert side, "side must be not None!"
        assert ord_type, "ord_type must be not None!"
        assert sz, "sz must be not None and not empty!"

        method = "POST"
        path = "/api/v5/trade/order"
        json = {
            "instId": inst_id,
            "tdMode": td_mode.value,
            "side": side.value,
            "posSide": posSide,
            "ordType": ord_type.value,
            "sz": sz,
            "px": px,
            "ccy": "USDT",
            "clOrdId": str(int(time.time() * 1000000)),
        }
        if client_ord_id is not None:
            json['clOrdId'] = client_ord_id
        response = self._query.query(method=method, path=path, json=json)
        return response.json()

    def cancel_order(self, inst_id: str, ord_id: str = None, cl_ord_id: str = None):
        assert inst_id, "inst_id must be not None and not empty!"
        assert ord_id or cl_ord_id, "either ord_id or cl_ord_id must be not None!"

        method = "POST"
        path = "/api/v5/trade/cancel-order"
        json = {
            "instId": inst_id,
            "ordId": ord_id,
            "clOrdId": cl_ord_id,
        }
        response = self._query.query(method=method, path=path, json=json)
        return response.json()

    def amend_order(
            self,
            inst_id: str,
            sz: str,
            px: str = None,
            ord_id: str = None,
            client_ord_id: str = None,
    ):
        assert inst_id, "inst_id must be not None and not empty!"
        assert sz, "sz must be not None and not empty!"

        method = "POST"
        path = "/api/v5/trade/amend-order"
        json = {
            "instId": inst_id,
            "newSz": sz,
            "newPx": px,
            "ordId": ord_id,
            "clOrdId": client_ord_id,
            "reqId": str(int(time.time() * 1000000)),
        }
        response = self._query.query(method=method, path=path, json=json)
        print(response.request.url)
        print(response.request.body)
        return response.json()

    def get_1line_k_line(self, bar: str, symbol: str):
        method = "GET"
        path = "/api/v5/market/candles"
        json = {
            'bar': bar,
            'instId': symbol,
            'limit': '10'
        }
        response = self._query.query(method=method, path=path, params=json)
        data = pd.DataFrame(response.json()['data'],
                            columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm']).sort_values(
            by=['ts'])
        data[['ts', 'confirm']] = data[['ts', 'confirm']].astype('int64')
        data[['o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote']] = data[
            ['o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote']].astype(float)
        return data.iloc[-1]

    def get_his_k_line(self, bar: str, symbol: str, after: int, before: int = 0):
        method = "GET"
        path = "/api/v5/market/history-candles"
        total_data = []
        start_time = deepcopy(before)
        # end_time = deepcopy(after)

        while after > start_time:
            try:
                json = {
                    'bar': bar,
                    'instId': symbol,
                    'limit': '300',
                    'after': str(after) if after != 0 else '',
                    'before': str(before) if before != 0 else '',
                }
                response = self._query.query(method=method, path=path, params=json)
                data = response.json()
                if response.status_code == 200 and 'data' in data and len(data['data']) > 0:
                    candles = data['data']
                    total_data.extend(candles)
                    after = int(candles[-1][0])
                else:
                    break
                time.sleep(0.1)
            except Exception as e:
                print(e)
        data = pd.DataFrame(total_data,
                            columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm']).sort_values(
            by=['ts'])
        data[['ts', 'confirm']] = data[['ts', 'confirm']].astype('int64')
        data[['o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote']] = data[
            ['o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote']].astype(float)

        return data

    def get_price_limit(self, symbol: str):
        method = "GET"
        path = "/api/v5/public/price-limit"
        json = {'instId': symbol}
        response = self._query.query(method=method, path=path, params=json)
        return {'buy_price': response.json()['data'][0]['buyLmt'], 'sell_price': response.json()['data'][0]['sellLmt']}


if __name__ == "__main__":
    from coin_python.utils import get_historical_k_line_local, get_today_zero_to_newest_timestamp
    # COIN_TRADING_KEY is env params in the LINUX system.
    # User can define it with your own
    okex = PrivateClient('your API key path', "Spot.Okex.v5")
    print(okex.get_price_limit('BTC-USDT-SWAP'))
    pre_20_days_his_data = get_historical_k_line_local('BTC-USDT-SWAP', datetime.today() - timedelta(20),
                                                       datetime.today() - timedelta(1))
    # # how to get today k line from 0 to this moment
    start_, end_ = get_today_zero_to_newest_timestamp()
    # okex = PrivateClient(os.getenv('COIN_TRADING_KEY'), "Spot.Okex.v5")
    today_k_bar = okex.get_his_k_line('1m', 'BTC-USDT-SWAP', end_, start_)
    print(today_k_bar.sort_values(by=['ts']))
    whole_k_lines = pd.concat([pre_20_days_his_data, today_k_bar]).sort_values(by=['ts'])
    k_bar_now = okex.get_1line_k_line('1m', 'BTC-USDT-SWAP').to_frame().T
    # whole_k_lines pops 1st data and put in k_bar_now
    print(k_bar_now)
    print(whole_k_lines)
