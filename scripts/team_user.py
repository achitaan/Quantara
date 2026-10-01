"""Create a team login interactively; no password is printed or passed on the command line."""

from getpass import getpass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from quantara.config import settings
from quantara.db import Store

if __name__ == "__main__":
    username = input("Username: ").strip()
    password = getpass("Password: ")
    if not username or len(password) < 12:
        raise ValueError("Use a username and a password of at least 12 characters")
    store = Store(settings.database_url)
    try:
        store.get("user", "user:" + username, "system")
    except KeyError:
        store.team_user(username, password)
        print("Created team user " + username)
    else:
        raise ValueError("User already exists; no credentials were changed")
