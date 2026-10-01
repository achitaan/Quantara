"""The sole supported backend entrypoint: uvicorn app:app --app-dir backend."""
from quantara.api import create_app

app = create_app()
