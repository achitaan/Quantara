import secrets

import pytest


@pytest.fixture(autouse=True)
def private_test_credentials(monkeypatch):
    monkeypatch.setenv("TEAM_USERNAME", "demo")
    monkeypatch.setenv("TEAM_PASSWORD", secrets.token_urlsafe(24))
