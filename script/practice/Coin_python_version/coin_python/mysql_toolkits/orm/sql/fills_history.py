from markdown_it.common.html_re import comment
from sqlalchemy import Column, INTEGER, VARCHAR, FLOAT, BIGINT, DATE, TIME, DATETIME, BOOLEAN, TEXT

from mysql_db.orm.sql import Base

Schema = 'coin'


class _FillsHistoryBase(object):
    __tablename__ = NotImplementedError
    __table_args__ = {'schema': Schema}

    Key = Column('key', BIGINT, primary_key=True, autoincrement=True, comment='自增主键')

    @classmethod
    def table_name(cls):
        return cls.__tablename__

    @classmethod
    def schema(cls):
        return cls.__table_args__['schema']


class OkexFillsHistory(_FillsHistoryBase, Base):
    __tablename__ = 'okex_fills_history'
    InstType = Column('instType', VARCHAR(10), index=True, comment='产品类型')
    InstId = Column('instId', VARCHAR(20), index=True, comment='产品ID')
    TradeId = Column('tradeId', VARCHAR(20), index=True, comment='最新成交ID')
    OrdId = Column('ordId', VARCHAR(20), index=True, comment='订单ID')
    ClOrdId = Column('clOrdId', VARCHAR(20), comment='用户自定义订单ID ')
    SubType = Column('subType', VARCHAR(20), index=True, comment='成交类型')
    FillPx = Column('fillPx', FLOAT, comment='最新成交价格')
    FillSz = Column('fillSz', FLOAT, comment='最新成交数量')
    Side = Column('side', VARCHAR(20), index=True, comment='订单方向 buy买 sell卖')
    PosSide = Column('posSide', VARCHAR(20), index=True, comment='持仓方向 long多 short空 买卖模式返回net')
    ExecType = Column('execType', VARCHAR(20), index=True, comment='流动性方向 T：taker M：maker')
    FeeCcy = Column('feeCcy', VARCHAR(20), index=True, comment='手续费币种')
    Fee = Column('fee', FLOAT, comment='手续费')
    Ts = Column('ts', BIGINT, comment='成交明细产生时间')
    FillTime = Column('fillTime', BIGINT, comment='成交时间')
    FeeRate = Column('feeRate', FLOAT, comment='手续费费率')


if __name__ == '__main__':

    from mysql_db.orm.sql import build_table
    from mysql_db.utils.conn import SQL

    eng = SQL.pick_engine('finance_database')

    for table, renew in [
        (OkexFillsHistory, True),
    ]:
        build_table(table, renew, eng=eng, verbose=True)
