from sqlalchemy import Integer, String
from sqlalchemy.sql.schema import Column
from .database import Base


class Users(Base):
    __tablename__ = 'User_Details'

    id = Column(Integer,primary_key=True)
    username = Column(String, nullable=False)
    email = Column(String, nullable=False)
    designation = Column(String, nullable=False)

