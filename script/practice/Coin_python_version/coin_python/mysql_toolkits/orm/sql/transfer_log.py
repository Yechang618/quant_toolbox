from sqlalchemy import Column, INTEGER, VARCHAR, FLOAT, BIGINT, DATE, TIME, DATETIME, BOOLEAN, TEXT

from mysql_db.orm.sql import Base

Schema = 'coin'


class _TransferLogBase(object):
    __tablename__ = NotImplementedError
    __table_args__ = {'schema': Schema}

    Key = Column('key', BIGINT, primary_key=True, autoincrement=True, comment='自增主键')

    @classmethod
    def table_name(cls):
        return cls.__tablename__

    @classmethod
    def schema(cls):
        return cls.__table_args__['schema']


class OKexTransferLog(_TransferLogBase, Base):
    __tablename__ = 'okex_transfer_log'
    AccountName = Column('account_name', VARCHAR(120), index=True, comment='账户名字')
    BillID = Column('bill_id', VARCHAR(255), index=True, comment='bill类型')
    BillType = Column('bill_type', VARCHAR(50), index=True, comment='bill类型')
    FillTime = Column('fill_time', BIGINT, index=True, comment='最近一次上传时间')
    BillSubType = Column('bill_sub_type', INTEGER, index=True, comment='bill子类型')
    BalanceAmountChange = Column('balance_amount_change', FLOAT, comment='账户层面balance变化量')
    Balance = Column('balance', FLOAT, comment='账户层面balance')
    Currency = Column('currency', VARCHAR(20), comment='账户balance币种')
    Fee = Column('fee', FLOAT, comment='手续费')
    PNL = Column('pnl', FLOAT, comment='损益')
    From = Column('from_side', VARCHAR(20), comment='付款方')
    To = Column('to_side', VARCHAR(20), comment='收款方')


if __name__ == '__main__':

    from mysql_db.orm.sql import build_table
    from mysql_db.utils.conn import SQL

    eng = SQL.pick_engine('finance_database')

    for table, renew in [
        (OKexTransferLog, True),
    ]:
        build_table(table, renew, eng=eng, verbose=True)
