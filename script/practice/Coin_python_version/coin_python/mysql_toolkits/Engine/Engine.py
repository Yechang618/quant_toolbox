import os

from contextlib import contextmanager

from sqlalchemy import MetaData, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Query, sessionmaker, Session
from sqlalchemy.pool import QueuePool

from mysql_toolkits.config.config import sql_con,global_configs,config_file

Base = declarative_base()
Meta = MetaData()

import sys
sys.path.append(os.path.join(os.getcwd()))


class QueryTool(Query):

    def __init__(self, entities, engine):
        session = sessionmaker(bind=engine)()
        super().__init__(entities, session)



def build_table(table, renew=False, eng=None, verbose=False):
    if eng is None:
        eng = EnginePointer.get_engine()

    table_info = f'{table.__table_args__["schema"]}.{table.__tablename__}'

    if table.__table__.exists(eng) and not renew:
        if verbose:
            print(f'{table_info} exits.')
    elif table.__table__.exists(eng) and renew:
        if verbose:
            print(f'{table_info} exits.')
        table.__table__.drop(eng)

        if verbose:
            print(f'{table_info} dropped.')
        table.__table__.create(eng)

        if verbose:
            print(f'{table_info} created.')
    else:
        table.__table__.create(eng)
        if verbose:
            print(f'{table_info} created.')


@contextmanager
def session_scope(eng=None):
    if eng is None:
        eng = EnginePointer.get_engine()
    session = Session(bind=eng)
    try:
        yield session
        session.commit()
    except Exception as _:
        session.rollback()
        raise
    finally:
        session.close()


def truncate_table(table, eng=None):
    if eng is None:
        session = sessionmaker(EnginePointer.get_engine())()
    else:
        session = sessionmaker(eng)()
    session.query(table).delete(synchronize_session='fetch')
    session.commit()
    session.close()


class EnginePointer(object):

    # _dev = None
    # _prod = None
    # _local_host = None
    # _prod_backup = None

    _label = None
    _engine = None
    # _session = None

    _default_eng_config = dict(
        max_overflow=30,
        pool_size=30,
        pool_timeout=10,
        poolclass=QueuePool,
        pool_recycle=7200,
        connect_args={'charset': 'utf8'}
    )

    @classmethod
    def dev(cls, **kwargs):
        if cls._dev is None:
            cls._dev = cls.get_engine('dev')
        return cls._dev

    @classmethod
    def renew(cls, engine_name=None, **kwargs):

        if cls._engine is not None:
            cls._engine.dispose()

        cls._engine = cls.picker(engine_name, **kwargs)
        cls._label = engine_name

    @classmethod
    def get_engine(cls, engine_name=None):
        if cls._engine is None or \
                cls._label is None or \
                (engine_name is not None and engine_name != cls._label) or \
                (os.getenv('OMK_DB_NAME') is not None and os.getenv('OMK_DB_NAME') != cls._label):
            cls.renew(engine_name)
        return cls._engine

    @classmethod
    def picker(cls, engine_name=None, **kwargs):

        if engine_name is None:
            if bool(os.getenv('OMK_DB_USE_ENV')) and os.getenv('OMK_DB'):
                engine_name = os.getenv('OMK_DB')
            else:
                engine_name = global_configs.get('eng_type')

        engine_config = sql_con.get(engine_name, None)
        if engine_config is None:
            raise ValueError('There is no config information for {!r}. Please set it up in {} first.'.format(
                engine_name, config_file
            ))

        host = engine_config.get('host', '')
        port = int(engine_config.get('port', 3306))
        user = engine_config.get('user', '')
        password = engine_config.get('password', '')
        db = engine_config.get('database', '')
        eng_kwargs = cls._default_eng_config.copy()
        eng_kwargs.update(kwargs)
        return create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}", **eng_kwargs)


if __name__=='__main__':
    engine=EnginePointer.picker('finance_database')
    print(engine.connect())