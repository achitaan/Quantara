from concurrent.futures import ThreadPoolExecutor
import os
from uuid import uuid4

import pytest

from quantara.db import Store


@pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason="PostgreSQL integration URL not configured",
)
def test_postgres_migrations_ownership_and_locked_updates():
    store = Store(os.environ["TEST_POSTGRES_URL"])
    owner = "test-" + str(uuid4())
    record = store.create("counter", owner, {"value": 0})

    def increment():
        independent = Store(os.environ["TEST_POSTGRES_URL"])
        with independent.edit("counter", record["id"], owner) as saved:
            saved["value"] += 1
        independent.engine.dispose()

    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(lambda _: increment(), range(12)))
        assert store.get("counter", record["id"], owner)["value"] == 12
        with pytest.raises(KeyError):
            store.get("counter", record["id"], "different-owner")
    finally:
        store.delete("counter", record["id"], owner)
        store.engine.dispose()
