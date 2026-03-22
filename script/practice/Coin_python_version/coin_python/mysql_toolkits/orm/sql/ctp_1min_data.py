from sqlalchemy import Column, VARCHAR, BIGINT, DATE, FLOAT, UniqueConstraint,DATETIME

from mysql_db.orm.sql import Base


Schema = 'ctp_data'

class _SchemaBase(object):
    __tablename__ = NotImplemented
    __table_args__ = {'schema': Schema}

    Key = Column('key', BIGINT, autoincrement=True, primary_key=True)

    @classmethod
    def table_name(cls):
        return cls.__tablename__

    @classmethod
    def schema(cls):
        return cls.__table_args__['schema']


class CTPTemp(_SchemaBase, Base):
    __tablename__ = 'ctp_1min_temp'
    SYMBOL =Column('symbol', VARCHAR(50), index=True, comment='合约标识')
    EXCHANGE = Column('exchange', VARCHAR(50), index=True, comment='交易所')
    DATETIME = Column('datetime',DATETIME, index=True, comment='交易时间')
    INTERVEL = Column('interval', VARCHAR(50), index=True, comment='数据频率')
    VOLUME = Column('volume', FLOAT, comment='交易量')
    TURNOVER=Column('turnover', FLOAT, comment='换手量')
    OPENINTEREST = Column('open_interest', FLOAT, comment='持仓量')
    OPENPRICE = Column('open_price', FLOAT, comment='开盘价')
    CLOSEPRICE = Column('close_price', FLOAT, comment='收盘价')
    HIGHPRICE = Column('high_price', FLOAT, comment='最高价')
    LOWPRICE = Column('low_price', FLOAT, comment='最低价')
    ASKONE=Column('askprice1',FLOAT,comment='ask最优价')
    BIDONE = Column('bidprice1', FLOAT, comment='bid最优价')

class CTPReal(_SchemaBase, Base):
    __tablename__ = 'dbbardata'
    SYMBOL =Column('symbol', VARCHAR(50), index=True, comment='合约标识')
    EXCHANGE = Column('exchange', VARCHAR(50), index=True, comment='交易所')
    DATETIME = Column('datetime',DATETIME, index=True, comment='交易时间')
    INTERVEL = Column('interval', VARCHAR(50), index=True, comment='数据频率')
    VOLUME = Column('volume', FLOAT, comment='交易量')
    TURNOVER=Column('turnover', FLOAT, comment='换手量')
    OPENINTEREST = Column('open_interest', FLOAT, comment='持仓量')
    OPENPRICE = Column('open_price', FLOAT, comment='开盘价')
    CLOSEPRICE = Column('close_price', FLOAT, comment='收盘价')
    HIGHPRICE = Column('high_price', FLOAT, comment='最高价')
    LOWPRICE = Column('low_price', FLOAT, comment='最低价')
    ASKONE=Column('askprice1',FLOAT,comment='ask最优价')
    BIDONE = Column('bidprice1', FLOAT, comment='bid最优价')

class CTPReal_Test(_SchemaBase, Base):
    __tablename__ = 'dbbardata_test'
    SYMBOL =Column('symbol', VARCHAR(50), index=True, comment='合约标识')
    EXCHANGE = Column('exchange', VARCHAR(50), index=True, comment='交易所')
    DATETIME = Column('datetime',DATETIME, index=True, comment='交易时间')
    INTERVEL = Column('interval', VARCHAR(50), index=True, comment='数据频率')
    VOLUME = Column('volume', FLOAT, comment='交易量')
    TURNOVER=Column('turnover', FLOAT, comment='换手量')
    OPENINTEREST = Column('open_interest', FLOAT, comment='持仓量')
    OPENPRICE = Column('open_price', FLOAT, comment='开盘价')
    CLOSEPRICE = Column('close_price', FLOAT, comment='收盘价')
    HIGHPRICE = Column('high_price', FLOAT, comment='最高价')
    LOWPRICE = Column('low_price', FLOAT, comment='最低价')
    ASKONE=Column('askprice1',FLOAT,comment='ask最优价')
    BIDONE = Column('bidprice1', FLOAT, comment='bid最优价')


if __name__=='__main__':
    from mysql_db.orm.sql import build_table
    from mysql_db.utils.conn import SQL

    eng = SQL.pick_engine('finance_database')

    for table, renew in [
        (CTPTemp,True),
        (CTPReal,True),
        (CTPReal_Test,True),
    ]:
        build_table(table, renew, eng=eng, verbose=True)