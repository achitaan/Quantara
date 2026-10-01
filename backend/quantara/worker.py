from celery import Celery

from .config import settings
from .db import Store
from .jobs import Jobs
from .service import Services

celery_app = Celery("quantara", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_schedule={"forward-paper": {"task": "quantara.poll_paper", "schedule": 60.0}},
)


@celery_app.task(name="quantara.execute")
def execute(identifier, owner):
    store = Store(settings.database_url)
    services = Services(store, settings)
    jobs = Jobs(store, services.execute, "celery", settings.redis_url)
    try:
        jobs.execute(identifier, owner)
    finally:
        jobs.close()
        store.engine.dispose()


@celery_app.task(name="quantara.poll_paper")
def poll_paper():
    store = Store(settings.database_url)
    try:
        for user in store.list("user", "system"):
            owner = user["username"]
            recovery = Jobs(
                store, Services(store, settings).execute, "celery", settings.redis_url
            )
            recovery.recover_stale(owner)
            recovery.close()
            for account in store.list("paper", owner):
                if account["status"] == "running" and account["mode"] == "forward":
                    # Do not enqueue another poll while one for this account is pending.
                    busy = any(
                        j["operation"] == "paper_poll"
                        and j["arguments"]["paper_id"] == account["id"]
                        and j["status"] in ("running", "queued")
                        for j in store.list("job", owner)
                    )
                    if not busy:
                        jobs = Jobs(
                            store,
                            Services(store, settings).execute,
                            "celery",
                            settings.redis_url,
                        )
                        jobs.submit("paper_poll", {"paper_id": account["id"]}, owner)
                        jobs.close()
    finally:
        store.engine.dispose()
