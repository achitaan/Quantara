import secrets

import pytest

from quantara.db import Store


def test_bootstrap_requires_private_password(monkeypatch):
    store = Store("sqlite://")
    monkeypatch.delenv("TEAM_PASSWORD")
    with pytest.raises(ValueError, match="TEAM_PASSWORD"):
        store.seed_user(True)
    with pytest.raises(ValueError, match="TEAM_PASSWORD"):
        store.seed_user(False)
    store.engine.dispose()


def test_rotation_revokes_only_target_user_sessions():
    store = Store("sqlite://")
    previous, replacement, other = (secrets.token_urlsafe(24) for _ in range(3))
    store.team_user("demo", previous)
    store.team_user("other", other)
    old_session = store.login("demo", previous)
    other_session = store.login("other", other)
    store.rotate_password("demo", replacement)
    with pytest.raises(ValueError, match="Invalid credentials"):
        store.login("demo", previous)
    with pytest.raises(KeyError):
        store.authenticate(old_session)
    assert store.authenticate(store.login("demo", replacement)) == "demo"
    assert store.authenticate(other_session) == "other"
    store.engine.dispose()
