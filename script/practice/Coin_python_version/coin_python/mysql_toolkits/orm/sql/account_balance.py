from sqlalchemy import Column, INTEGER, VARCHAR, FLOAT, BIGINT, DATE, TIME, DATETIME, BOOLEAN, TEXT

from mysql_db.orm.sql import Base

Schema = 'coin'


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
    __tablename__ = 'okex_account_balance'
    Ccy = Column('ccy', VARCHAR(10), index=True, comment='币种')
    CashBal = Column('cashBal', FLOAT, comment='币种余额')
    AvailBal = Column('availBal', FLOAT, comment='可用余额')
    AvailEq = Column('availEq', FLOAT, comment='可用保证金')
    UTime = Column('uTime', BIGINT, comment='余额最新时间')
    FrozenBal = Column('frozenBal', FLOAT, comment='币种占用金额')
    OrdFrozen = Column('ordFrozen', INTEGER, comment='挂单冻结数量')
    MaxLoan = Column('maxLoan', INTEGER, comment='币种最大可借数量')
    SpotBal = Column('spotBal', FLOAT, comment='现货余额')
    OpenAvgPx = Column('openAvgPx', FLOAT, comment='现货开仓成本价')
    AccAvgPx = Column('accAvgPx', FLOAT, comment='现货累计成本价')
    TotalPnl = Column('totalPnl', FLOAT, comment='现货累计收益')


if __name__ == '__main__':

    from mysql_db.orm.sql import build_table
    from mysql_db.utils.conn import SQL

    eng = SQL.pick_engine('finance_database')

    for table, renew in [
        (OkexAccountBalance, True),
    ]:
        build_table(table, renew, eng=eng, verbose=True)
