from sqlalchemy import Column, INTEGER, VARCHAR, FLOAT, BIGINT, DATE, TIME, DATETIME, BOOLEAN, TEXT

from mysql_db.orm.sql import Base

Schema = 'futures'


class _BalanceBase(object):
    __tablename__ = NotImplementedError
    __table_args__ = {'schema': Schema}

    Key = Column('key', BIGINT, primary_key=True, autoincrement=True, comment='自增主键')

    @classmethod
    def table_name(cls):
        return cls.__tablename__

    @classmethod
    def schema(cls):
        return cls.__table_args__['schema']


class OkexAccountBalance(_BalanceBase, Base):
    __tablename__ = 'futures_list'
    Userid = Column('user_id', VARCHAR(30), index=True, comment='用户ID')
    Orderid = Column('order_id', VARCHAR(30), comment='订单ID')
    Tradeid = Column('trade_id', VARCHAR(30), comment='交易ID')
    Commission = Column('commission', FLOAT, comment='手续费')
    Direction = Column('direction', VARCHAR(10), comment='交易方向')
    Exchangeid = Column('exchange_id', VARCHAR(30), comment='交易所ID')
    Instrumentid = Column('instrument_id', VARCHAR(10), comment='合约代码')
    Price = Column('price', FLOAT, comment='交易价格')
    Tradedatetime = Column('trade_date_time', DATETIME, comment='交易时间')
    Userid = Column('user_id', VARCHAR(10), comment='用户ID')
    Volume = Column('volume', INTEGER, comment='交易量')

if __name__ == '__main__':

    from mysql_db.orm.sql import build_table
    from mysql_db.utils.conn import SQL

    eng = SQL.pick_engine('finance_database')

    for table, renew in [
        (OkexAccountBalance, True),
    ]:
        build_table(table, renew, eng=eng, verbose=True)
