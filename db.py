"""
Relational storage: page sync state, per-query audit log, eval run history.
Deliberately NOT the vector store - Chroma owns embeddings, this owns
everything that's actually relational. Works identically against SQLite
(default) or Postgres (e.g. Neon) - same models, same code, only
DATABASE_URL changes.
"""
import datetime as dt
import json
import uuid

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, create_engine, inspect, select
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> dt.datetime:
    return dt.datetime.utcnow()


class PageVersion(Base):
    """Tracks the last-synced Confluence version per page, for incremental sync."""

    __tablename__ = "page_versions"

    page_id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    url = Column(String, nullable=False)
    synced_at = Column(DateTime, default=_now, onupdate=_now)
    deleted_at = Column(DateTime, nullable=True)  # set once the page is gone from Confluence


class QueryAuditLog(Base):
    """One row per question the app answered. This is the audit trail."""

    __tablename__ = "query_audit_log"

    id = Column(String, primary_key=True, default=_uuid)
    question = Column(Text, nullable=False)
    router_decision = Column(String, nullable=False)  # in_domain | needs_external | out_of_domain
    router_confidence = Column(Float, nullable=True)
    chunk_ids = Column(JSON, nullable=True)  # list[str], the chunks actually used
    groundedness_verdict = Column(String, nullable=True)  # supported | partial | unsupported | n/a
    answer = Column(Text, nullable=False)
    source_type = Column(String, nullable=False)  # internal | external | refused
    latency_ms = Column(Integer, nullable=True)
    ts = Column(DateTime, default=_now)


class EvalRun(Base):
    """One row per Ragas evaluation run - lets eval quality be tracked over time,
    not just pass/fail on the latest commit."""

    __tablename__ = "eval_runs"

    id = Column(String, primary_key=True, default=_uuid)
    run_ts = Column(DateTime, default=_now)
    git_sha = Column(String, nullable=True)
    num_questions = Column(Integer, nullable=False)
    context_precision = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    raw_results = Column(JSON, nullable=True)  # per-question breakdown


_engine = create_engine(settings.database_url, future=True)
Session = sessionmaker(bind=_engine, future=True)


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup.

    Also covers the one column added after the table already shipped
    (page_versions.deleted_at) - create_all() only creates missing tables,
    it never alters existing ones, and there's no migration framework here.
    """
    Base.metadata.create_all(_engine)
    inspector = inspect(_engine)
    if "page_versions" in inspector.get_table_names():
        cols = {c["name"] for c in inspector.get_columns("page_versions")}
        if "deleted_at" not in cols:
            col_type = "TIMESTAMP" if _engine.dialect.name != "sqlite" else "DATETIME"
            with _engine.connect() as conn:
                conn.exec_driver_sql(f"ALTER TABLE page_versions ADD COLUMN deleted_at {col_type}")
                conn.commit()


def log_query(
    *,
    question: str,
    router_decision: str,
    router_confidence: float | None,
    chunk_ids: list[str],
    groundedness_verdict: str | None,
    answer: str,
    source_type: str,
    latency_ms: int | None = None,
) -> None:
    with Session() as session:
        session.add(
            QueryAuditLog(
                question=question,
                router_decision=router_decision,
                router_confidence=router_confidence,
                chunk_ids=chunk_ids,
                groundedness_verdict=groundedness_verdict,
                answer=answer,
                source_type=source_type,
                latency_ms=latency_ms,
            )
        )
        session.commit()


def get_synced_version(page_id: str) -> int | None:
    with Session() as session:
        row = session.get(PageVersion, page_id)
        return row.version if row else None


def upsert_page_version(*, page_id: str, title: str, version: int, url: str) -> None:
    with Session() as session:
        row = session.get(PageVersion, page_id)
        if row:
            row.title, row.version, row.url = title, version, url
        else:
            row = PageVersion(page_id=page_id, title=title, version=version, url=url)
            session.add(row)
        session.commit()


def get_all_page_ids() -> list[str]:
    """Ids of pages we currently consider live (not yet tombstoned) - used by
    reindex.py to detect pages that vanished from Confluence's page list."""
    with Session() as session:
        rows = session.scalars(
            select(PageVersion.page_id).where(PageVersion.deleted_at.is_(None))
        )
        return list(rows)


def mark_page_deleted(page_id: str) -> None:
    with Session() as session:
        row = session.get(PageVersion, page_id)
        if row and row.deleted_at is None:
            row.deleted_at = _now()
            session.commit()


def record_eval_run(
    *,
    num_questions: int,
    context_precision: float,
    context_recall: float,
    faithfulness: float,
    answer_relevancy: float,
    raw_results: list[dict],
    git_sha: str | None = None,
) -> None:
    with Session() as session:
        session.add(
            EvalRun(
                num_questions=num_questions,
                context_precision=context_precision,
                context_recall=context_recall,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                raw_results=json.loads(json.dumps(raw_results, default=str)),
                git_sha=git_sha,
            )
        )
        session.commit()
