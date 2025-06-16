from sqlalchemy import create_engine, Engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import URL

from backend.config.db_config import POSTGRESQL_DRIVER_NAME


class DbInitializer:
    def __init__(self, driver_name:str, db_host_name:str, db_name:str, db_user_name:str, db_password:str, db_port: int):
        self.driver_name = driver_name
        self.db_host_name = db_host_name
        self.db_name = db_name
        self.db_user_name = db_user_name
        self.db_password = db_password
        self.db_port = db_port
        self.engine = None
        self.session = None

    def db_create_engin(self)->Engine:
        url_object = URL.create(
            POSTGRESQL_DRIVER_NAME,
            username=self.db_user_name,
            password=self.db_password,
            host=self.db_host_name,
            database=self.db_name,)
        self.engine = create_engine(url_object)
        return self.engine

    def db_create_session(self)->sessionmaker:
        self.session = sessionmaker(bind=self.engine)
        return self.session