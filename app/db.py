"""This file is used for defining the database connection"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///./database.sqlite", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@event.listens_for(engine, "connect")
def _enable_sqlite_wal(dbapi_connection, connection_record) -> None:
    """Reduce writer lock contention for predict-request logging under light concurrency."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class PredictRequestLog(Base):
    __tablename__ = "predict_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    passenger_id = Column(Integer, nullable=False, index=True)
    recency_7 = Column(Integer, nullable=False)
    frequency_7 = Column(Integer, nullable=False)
    monetary_7 = Column(Float, nullable=False)

    prediction_monetary_30 = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
