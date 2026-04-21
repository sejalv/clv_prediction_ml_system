"""
This file gives you some shared resources for your tests.
You could also add things that should be executed on startup here (e.g database related things).
"""

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.db import Base, SessionLocal


@pytest.fixture(scope="module")
def client() -> Generator:
    """
    TestClient that:
    - replaces the real sklearn model with a MagicMock returning a fixed scalar (42.0)
    - uses an isolated in-memory SQLite DB so tests never accumulate state
    """
    # Override DB tables to use an isolated in-memory engine for tests
    import sqlalchemy
    from sqlalchemy.pool import StaticPool

    test_engine = sqlalchemy.create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=test_engine)
    TestSession = sqlalchemy.orm.sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    from app.db import get_db

    app.dependency_overrides[get_db] = override_get_db

    mock_model = MagicMock()
    mock_model.predict.return_value = [42.0]

    with patch.object(app.state, "model", mock_model, create=True), patch.object(
        app.state, "model_version", "test-version", create=True
    ):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def db() -> Generator:
    yield SessionLocal()
