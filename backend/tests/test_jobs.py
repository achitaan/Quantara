from threading import Event
import time

from quantara.db import Store, now
from quantara.jobs import Cancelled, Jobs


def test_cancelled_attempt_cannot_overwrite_retry(tmp_path):
    store = Store("sqlite:///" + str(tmp_path / "jobs.db"))
    started, release, settled = Event(), Event(), Event()
    calls = []

    def runner(operation, arguments, owner, progress):
        calls.append(True)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
            try:
                progress(0.5, "Stale attempt")
            except Cancelled:
                settled.set()
            return {"value": "stale"}
        return {"value": "retry"}

    jobs = Jobs(store, runner)
    try:
        job = jobs.submit("slow", {}, "team")
        assert started.wait(5)
        with store.edit("job", job["id"], "team") as saved:
            saved.update(status="cancelled", updated_at=now())
        jobs.retry(job["id"], "team")
        for _ in range(200):
            saved = store.get("job", job["id"], "team")
            if saved["status"] == "complete":
                break
            time.sleep(0.01)
        assert saved["result"] == {"value": "retry"}
        release.set()
        assert settled.wait(5)
        assert store.get("job", job["id"], "team")["result"] == {"value": "retry"}
    finally:
        release.set()
        jobs.close()
