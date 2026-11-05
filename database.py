import os
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

DATABASE_URL = (
    os.getenv("PROJECTS_DATABASE_URL")
    or os.getenv("APP_DATABASE_URL")
    or "sqlite:///./projects.db"
)

engine_kwargs = {"connect_args": {"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class ProjectGeneration(Base):
    __tablename__ = "project_generations"

    id = Column(String, primary_key=True, index=True, default=lambda: uuid.uuid4().hex)
    prompt = Column(Text, nullable=False)
    status = Column(String, default="pending")  # pending, running, done, failed, rolled_back
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime, 
        default=lambda: datetime.now(timezone.utc), 
        onupdate=lambda: datetime.now(timezone.utc)
    )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database() -> None:
    """Ensure core tables exist and DB is reachable."""
    Base.metadata.create_all(bind=engine)


def verify_database_health() -> bool:
    """Run a minimal health query; return False on any failure."""
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except Exception:
        return False


@contextmanager
def transactional_session() -> Session:
    """Provide a transaction scope with rollback-on-error."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
