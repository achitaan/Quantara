"""Versioned storage shared by API and Celery; JSON payloads have typed API schemas."""

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import pbkdf2_hmac, sha256
import hmac
import os
import secrets
from threading import RLock
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Column,
    Integer,
    String,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class Record(Base):
    __tablename__ = "records"
    id = Column(String(80), primary_key=True)
    kind = Column(String(40), nullable=False, index=True)
    owner = Column(String(100), nullable=False, index=True)
    payload = Column(JSON, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    created_at = Column(
        String(40),
        default=lambda: datetime.now(timezone.utc).isoformat(),
        nullable=False,
    )


class Store:
    def __init__(self, url):
        if url.startswith(("postgresql://", "postgres://")):
            url = "postgresql+psycopg://" + url.split("://", 1)[1]
        self.lock = RLock()
        self.engine = create_engine(
            url,
            connect_args={"check_same_thread": False}
            if url.startswith("sqlite")
            else {},
            pool_pre_ping=True,
            **(
                {"poolclass": StaticPool}
                if url in ("sqlite://", "sqlite:///:memory:")
                else {}
            ),
        )
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        self.migrate()

    def migrate(self):
        # Append numbered migrations; preserve existing records.
        with self.engine.begin() as conn:
            if self.engine.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(81736421)"))
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)"
                )
            )
            if not conn.execute(
                text("SELECT version FROM schema_migrations WHERE version=1")
            ).first():
                Base.metadata.create_all(conn)
                conn.execute(text("INSERT INTO schema_migrations (version) VALUES (1)"))
            if not conn.execute(
                text("SELECT version FROM schema_migrations WHERE version=2")
            ).first():
                columns = {
                    column["name"] for column in inspect(conn).get_columns("records")
                }
                if "created_at" not in columns:
                    conn.execute(
                        text(
                            "ALTER TABLE records ADD COLUMN created_at VARCHAR(40) NOT NULL DEFAULT '1970-01-01T00:00:00+00:00'"
                        )
                    )
                conn.execute(text("INSERT INTO schema_migrations (version) VALUES (2)"))

    def create(self, kind, owner, payload, identifier=None):
        identifier = identifier or str(uuid4())
        with self.lock, self.sessions.begin() as session:
            if session.get(Record, identifier):
                raise ValueError("Identifier already exists")
            session.add(Record(id=identifier, kind=kind, owner=owner, payload=payload))
        return {"id": identifier, **payload}

    def get(self, kind, identifier, owner):
        with self.sessions() as session:
            row = session.get(Record, identifier)
            if not row or row.kind != kind or row.owner != owner:
                raise KeyError("Record not found")
            return {"id": row.id, **row.payload}

    def list(self, kind, owner):
        with self.sessions() as session:
            rows = session.scalars(
                select(Record)
                .where(Record.kind == kind, Record.owner == owner)
                .order_by(Record.created_at, Record.id)
            ).all()
            return [{"id": r.id, **r.payload} for r in rows]

    @contextmanager
    def edit(self, kind, identifier, owner):
        # FOR UPDATE provides cross-worker serialization on PostgreSQL.
        with self.lock, self.sessions.begin() as session:
            row = session.scalar(
                select(Record)
                .where(
                    Record.id == identifier, Record.kind == kind, Record.owner == owner
                )
                .with_for_update()
            )
            if not row:
                raise KeyError("Record not found")
            value = dict(row.payload)
            yield value
            row.payload = value
            row.version += 1

    def delete(self, kind, identifier, owner):
        with self.lock, self.sessions.begin() as session:
            row = session.get(Record, identifier)
            if not row or row.kind != kind or row.owner != owner:
                raise KeyError("Record not found")
            session.delete(row)

    def team_user(self, username, password):
        ident = "user:" + username
        try:
            return self.get("user", ident, "system")
        except KeyError:
            salt = secrets.token_hex(16)
            digest = pbkdf2_hmac(
                "sha256", password.encode(), salt.encode(), 600000
            ).hex()
            return self.create(
                "user",
                "system",
                {"username": username, "salt": salt, "digest": digest},
                ident,
            )

    def login(self, username, password):
        try:
            user = self.get("user", "user:" + username, "system")
        except KeyError:
            # Keep unknown user attempts on the same expensive hashing path.
            user = {"salt": "unknown", "digest": "0" * 64}
        digest = pbkdf2_hmac(
            "sha256", password.encode(), user["salt"].encode(), 600000
        ).hex()
        if not hmac.compare_digest(digest, user["digest"]):
            raise ValueError("Invalid credentials")
        token = secrets.token_urlsafe(32)
        self.create(
            "session",
            "system",
            {
                "username": username,
                "expires": datetime.now(timezone.utc).timestamp() + 86400,
            },
            "session:" + sha256(token.encode()).hexdigest(),
        )
        return token

    def authenticate(self, token):
        if not token:
            raise KeyError("Authentication required")
        value = self.get(
            "session", "session:" + sha256(token.encode()).hexdigest(), "system"
        )
        if value["expires"] < datetime.now(timezone.utc).timestamp():
            raise KeyError("Session expired")
        return value["username"]

    def rotate_password(self, username, password):
        """Replace an existing user's password and revoke that user's sessions."""
        salt = secrets.token_hex(16)
        digest = pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600000).hex()
        with self.lock, self.sessions.begin() as session:
            row = session.get(Record, "user:" + username)
            if not row or row.kind != "user":
                raise KeyError("User not found")
            row.payload = {**row.payload, "salt": salt, "digest": digest}
            row.version += 1
            for saved in session.scalars(
                select(Record).where(Record.kind == "session")
            ):
                if saved.payload.get("username") == username:
                    session.delete(saved)

    def seed_user(self, demo):
        username = os.getenv("TEAM_USERNAME", "demo" if demo else "")
        password = os.getenv("TEAM_PASSWORD", "")
        if not username or len(password) < 12:
            raise ValueError(
                "Configure TEAM_USERNAME and TEAM_PASSWORD (at least 12 characters); "
                "run scripts/setup_local.py for local credentials"
            )
        self.team_user(username, password)


def now():
    return datetime.now(timezone.utc).isoformat()
