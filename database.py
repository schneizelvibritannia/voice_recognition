from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from settings import SQLALCHEMY_DATABASE_URL
import databases
from services import create_table

database_URL = databases.Database(SQLALCHEMY_DATABASE_URL)
metadata = MetaData()

user = create_table(metadata)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, pool_size=3, max_overflow=0
)
metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()