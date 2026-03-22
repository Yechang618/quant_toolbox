import aiomysql
import asyncio

import time

import pandas as pd


class Asyncio_MYQAL_Engine(object):
    def __init__(self, host_, port_, user_, password_, db_, loop):
        self._host = host_
        self._port = port_
        self._user = user_
        self._password = password_
        self._db = db_
        self._loop = loop

    async def singal_search_query(self, query, name):
        self._connection = await aiomysql.connect(host=self._host, port=self._port, user=self._user,
                                                  password=self._password, db=self._db,
                                                  loop=self._loop)
        async with self._connection.cursor() as cur:
            start_ = time.time()
            count = await cur.execute(query)
            end_ = time.time()
            if count:
                result_ = await cur.fetchall()
                cols_ = [x[0] for x in cur.description]
                return [result_, cols_, name]
            else:
                print('no data returned!')
                return None, None, query

    async def _pool_singal_search_query(self, pool, query, name=''):
        async with pool.acquire() as conn:
            start_ = time.time()
            async with conn.cursor() as cur:
                count = await cur.execute(query)
                end_ = time.time()
                if count:
                    result_ = await cur.fetchall()
                    cols_ = [x[0] for x in cur.description]
                    return [result_, cols_, name]
                else:
                    print(f'{query} no data returned!')
                    return [None, None, '']

    async def _many_singal_search(self, pool, query_list, name_list):
        return await asyncio.gather(
            *[self._pool_singal_search_query(pool, query, name) for query, name in zip(query_list, name_list)])

    async def many_search_exeuctor(self, query_list, name_ilst, min_link=1, max_link=5):
        async with aiomysql.create_pool(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                db=self._db,
                minsize=min_link,
                maxsize=max_link,
                echo=True,
                autocommit=True,
                loop=self._loop
        ) as pool:
            return await self._many_singal_search(pool, query_list, name_ilst)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    engine_ = Asyncio_MYQAL_Engine('192.168.3.18', 3306, 'root', '123', 'qtdb', loop)

    # searched_data = loop.run_until_complete(
    #     engine_.singal_search_query("select * from hq_stock_daily where trade_date>'2023-01-01'",''))
    # print(searched_data)
    name_list = ['cu_futures_data']
    month_list1 = [
        '2020-01-1', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01',
        '2020-09-01', '2020-10-01', '2020-11-01', '2020-12-01',
        '2021-01-1', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01',
        '2021-09-01', '2021-10-01', '2021-11-01', '2021-12-01',
        '2022-01-1', '2022-02-01', '2022-03-01', '2022-04-01', '2022-05-01',
    ]
    month_list2 = pd.period_range('2020-01-01','2022-05-01',freq='1m')
    month_list2 = [x.strftime('%Y-%m-%d') for x in month_list2]
    for m1, m2 in zip(month_list1, month_list2):
        query_list = [
            f'select * from hq_fut_min_1min where trade_date>="{m1}" and trade_date<="{m2}"'
        ]  # 2021-01-01～2022-05-01
        results = loop.run_until_complete(engine_.many_search_exeuctor(query_list, name_list))
        from local_file_api.utils import organise_data_to_df

        organise_data_to_df(results[0]).iloc[:, 0:2].to_feather(f'/mnt/项目中间数据保存处_缓冲/{m1}_{m2}.feather')
