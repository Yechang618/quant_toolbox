from sqlalchemy import Column, VARCHAR, BIGINT, DATE, FLOAT, UniqueConstraint

from mysql_db.orm.sql import Base

Schema = 'qtdb'


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


class TradingDate(_SchemaBase, Base):
    __tablename__ = 'hq_trade_cal'
    TradingDate = Column('cal_date', DATE, comment='交易日期', index=True)
    Exchange = Column('exchange', VARCHAR(32), index=True)
    IsOpen = Column('is_open', BIGINT, index=True)


if __name__ == '__main__':
    from mysql_db.orm.sql import build_table
    from mysql_db.utils.conn import SQL

    eng = SQL.pick_engine('finance_database')

    for table, renew in [
        (TradingDate, True),
    ]:
        build_table(table, renew, eng=eng, verbose=True)
