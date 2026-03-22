# -*- coding: utf-8 -*-
from sqlalchemy import Column, VARCHAR, BIGINT, DATE, INTEGER, FLOAT, DATETIME, BOOLEAN, TIME, DECIMAL, SMALLINT, \
    TEXT, func, UniqueConstraint

from mysql_db.orm.sql import Base

Schema = 'coin'


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


class OKexPosition(_SchemaBase, Base):
    __tablename__ = 'okex_position'
    AccountName = Column('account_name', VARCHAR(120), index=True, comment='账户名字')
    MarketType = Column('market_type', VARCHAR(50), index=True, comment='市场种类')
    Underlying = Column('underlying', VARCHAR(50), index=True, comment='品种')
    Position = Column('position', INTEGER, comment='持仓数量')
    Currency = Column('currency', VARCHAR(50), comment='占用保证金币种')
    Direction = Column('direction', VARCHAR(30), comment='持仓方向')
    Fee = Column('fee', FLOAT, comment='手续费')
    LastPrice = Column('last_price', FLOAT, comment='持仓信息更新时最新价格')
    PNL = Column('pnl', FLOAT, comment='平仓累计损益')
    RealisedPNL = Column('real_pnl', FLOAT, comment='已实现收益')
    UnrealisedPNL = Column('unreal_pnl', FLOAT, comment='未实现损益')
    AlgoID = Column('algo_id', VARCHAR(200), index=True, comment='策略下单ID')
    CreateTime = Column('create_time', DATETIME, index=True, comment='持仓创建时间')
    UpdateTime = Column('update_time', DATETIME, index=True, comment='持仓信息更新时间')




if __name__ == '__main__':

    from mysql_db.orm.sql import build_table
    from mysql_db.utils.conn import SQL

    eng = SQL.pick_engine('finance_database')

    for table, renew in [
        (OKexPosition, True),
    ]:
        build_table(table, renew, eng=eng, verbose=True)
