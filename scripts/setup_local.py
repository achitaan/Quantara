"""Generate ignored local credentials; optionally rotate an existing native login."""

import argparse
from pathlib import Path
import secrets
import sys

from dotenv import dotenv_values, set_key

ROOT = Path(__file__).resolve().parents[1]


def configure(rotate=False):
    target = ROOT / ".env"
    if not target.exists():
        target.write_text((ROOT / "backend/.env.example").read_text(), encoding="utf-8")
    values = dotenv_values(target)
    if values.get("TEAM_PASSWORD") and not rotate:
        raise ValueError(
            "Configuration exists; use --rotate-login to replace the native login."
        )
    credential = secrets.token_urlsafe(32)
    set_key(str(target), "TEAM_PASSWORD", credential)
    if not values.get("POSTGRES_PASSWORD"):
        # Hex is safe to interpolate in a database URL without URL encoding.
        set_key(str(target), "POSTGRES_PASSWORD", secrets.token_hex(32))
    if rotate:
        sys.path.insert(0, str(ROOT / "backend"))
        from quantara.config import settings
        from quantara.db import Store

        store = Store(settings.database_url)
        username = values.get("TEAM_USERNAME") or "demo"
        try:
            store.rotate_password(username, credential)
        except KeyError:
            store.team_user(username, credential)
        finally:
            store.engine.dispose()
    print("Private credentials saved in .env. Passwords were not printed.")
    print(
        "Restart the API; existing sessions are revoked when rotating a native login."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotate-login", action="store_true")
    configure(parser.parse_args().rotate_login)
