from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from uuid import uuid4

from .db import now

log = logging.getLogger(__name__)
current_job = ContextVar("quantara_job", default=None)


def resource_id(operation, arguments):
    job = current_job.get()
    if job is None:
        return None
    digest = sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()[:12]
    return operation + ":" + job + ":" + digest


class Cancelled(Exception):
    pass


class Jobs:
    def __init__(self, store, runner, backend="local", redis_url=None):
        self.store, self.runner, self.backend, self.redis_url = (
            store,
            runner,
            backend,
            redis_url,
        )
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="quantara")
        self.chat_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quantara-chat")

    def submit(self, operation, arguments, owner):
        job = self.store.create(
            "job",
            owner,
            {
                "operation": operation,
                "arguments": arguments,
                "status": "queued",
                "progress": 0,
                "message": "Queued",
                "created_at": now(),
                "updated_at": now(),
                "result": None,
                "error": None,
                "details": {},
            },
        )
        if self.backend == "celery":
            from .worker import celery_app

            celery_app.send_task("quantara.execute", args=[job["id"], owner])
        else:
            (self.chat_executor if operation == "chat" else self.executor).submit(self.execute, job["id"], owner)
        return job

    def execute(self, identifier, owner):
        attempt = uuid4().hex
        with self.store.edit("job", identifier, owner) as job:
            if job["status"] != "queued":
                return
            job.update(
                status="running", attempt=attempt, updated_at=now(), message="Starting"
            )

        def progress(fraction, message):
            with self.store.edit("job", identifier, owner) as saved:
                if saved["status"] != "running" or saved.get("attempt") != attempt:
                    raise Cancelled()
                saved.update(
                    progress=float(fraction), message=message, updated_at=now()
                )

        def snapshot(details):
            with self.store.edit("job", identifier, owner) as saved:
                if saved["status"] != "running" or saved.get("attempt") != attempt:
                    raise Cancelled()
                saved.update(details=details, updated_at=now())

        progress.snapshot = snapshot

        try:
            token = current_job.set(identifier)
            job = self.store.get("job", identifier, owner)
            result = self.runner(job["operation"], job["arguments"], owner, progress)
            with self.store.edit("job", identifier, owner) as saved:
                if saved["status"] != "running" or saved.get("attempt") != attempt:
                    return
                saved.update(
                    status="complete",
                    progress=1,
                    result=result,
                    message="Complete",
                    updated_at=now(),
                )
        except Cancelled:
            pass
        except Exception as exc:
            log.exception("Job %s failed", identifier)
            error = (
                str(exc)
                if isinstance(exc, (ValueError, KeyError, ImportError))
                else f"{type(exc).__name__}: operation failed; check server logs"
            )
            with self.store.edit("job", identifier, owner) as saved:
                if saved["status"] == "running" and saved.get("attempt") == attempt:
                    saved.update(
                        status="failed", error=error, message="Failed", updated_at=now()
                    )
        finally:
            current_job.reset(token)

    def retry(self, identifier, owner):
        with self.store.edit("job", identifier, owner) as job:
            if job["status"] not in ("failed", "cancelled"):
                raise ValueError("Only failed or cancelled jobs can be retried")
            job.update(
                status="queued",
                error=None,
                message="Queued for retry",
                    details={},
                updated_at=now(),
            )
        if self.backend == "celery":
            from .worker import celery_app

            celery_app.send_task("quantara.execute", args=[identifier, owner])
        else:
            (self.chat_executor if job["operation"] == "chat" else self.executor).submit(self.execute, identifier, owner)
        return self.store.get("job", identifier, owner)

    def recover_stale(self, owner, seconds=900):
        for job in self.store.list("job", owner):
            age = (
                datetime.now(timezone.utc) - datetime.fromisoformat(job["updated_at"])
            ).total_seconds()
            if job["status"] == "running" and age > seconds:
                with self.store.edit("job", job["id"], owner) as saved:
                    saved.update(
                        status="failed",
                        error="Worker lease expired. Retry this job.",
                        message="Interrupted",
                        updated_at=now(),
                    )

    def recover(self, owner):
        # Interrupted local tasks are explicitly retryable, never silently rerun with side effects.
        for job in self.store.list("job", owner):
            if job["status"] in ("queued", "running"):
                with self.store.edit("job", job["id"], owner) as saved:
                    saved.update(
                        status="failed",
                        error="Interrupted by restart. Retry this operation.",
                        message="Interrupted",
                        updated_at=now(),
                    )

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.chat_executor.shutdown(wait=False, cancel_futures=True)
