import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd

from mysql_db.utils.conn import SQL
from mysql_db.orm.sql.ctp_1min_data import CTPTemp, CTPReal,CTPReal_Test
from sqlalchemy.orm import sessionmaker, Query
from sqlalchemy import func


def _get_session(engine_name):
    engine_ = SQL.pick_engine(engine_name)
    session = sessionmaker(engine_)()
    return session, engine_


def get_ctp_latest_bar(engine_name, table_object):
    sess_, engine_ = _get_session(engine_name)
    max_date = Query(func.max(table_object.DATETIME), sess_).one_or_none()[0]
    return pd.read_sql(Query(table_object).filter(table_object.DATETIME == max_date).statement, con=engine_)


def get_ctp_temp_latest_time(engine_name):
    sess_, engine_ = _get_session(engine_name)
    return Query(func.max(CTPTemp.DATETIME), sess_).one_or_none()[0]


def get_ctp_real_latest_time(engine_name,debug=False):
    sess_, engine_ = _get_session(engine_name)
    if not debug:
        return Query(func.max(CTPReal.DATETIME), sess_).one_or_none()[0]
    else:
        return Query(func.max(CTPReal_Test.DATETIME), sess_).one_or_none()[0]


if __name__ == '__main__':
    print(get_ctp_latest_bar('finance_database', CTPTemp))
    bar_time = get_ctp_temp_latest_time('finance_database')
    print(bar_time)
