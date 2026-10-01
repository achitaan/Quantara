"""Record public Bluesky cashtag posts from Jetstream with replay-safe IDs."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from quantara.config import settings
from quantara.db import Store
from quantara.news import ingest
from quantara.schemas import NewsItem


async def main(owner, symbols, maximum, classifier):
    import websockets

    store = Store(settings.database_url)
    state_id = "social-cursor:" + owner
    try:
        cursor = store.get("social_cursor", state_id, owner)["cursor"]
    except KeyError:
        cursor = 0
        store.create("social_cursor", owner, {"cursor": 0}, state_id)
    count, delay = 0, 1
    while count < maximum:
        url = "wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"
        if cursor:
            url += "&cursor=" + str(cursor)
        try:
            async with websockets.connect(
                url, max_size=1_000_000, open_timeout=30
            ) as socket:
                delay = 1
                async for message in socket:
                    event = json.loads(message)
                    cursor = event["time_us"]
                    commit = event.get("commit", {})
                    identifier = (
                        "at://"
                        + event.get("did", "")
                        + "/app.bsky.feed.post/"
                        + commit.get("rkey", "")
                    )
                    if commit.get("operation") == "delete":
                        for old in store.list("news", owner):
                            if old["external_id"] == identifier:
                                store.delete("news", old["id"], owner)
                    elif commit.get("operation") == "create":
                        post = commit.get("record", {})
                        text = post.get("text", "")
                        tags = sorted(
                            set(re.findall(r"\$([A-Z][A-Z0-9.]{0,9})\b", text))
                            & set(symbols)
                        )
                        if tags:
                            item = NewsItem(
                                external_id=identifier,
                                timestamp=post["createdAt"],
                                available_at=datetime.now(timezone.utc),
                                symbols=tags,
                                text=text[:5000],
                                source="bluesky-public-jetstream",
                                url=identifier,
                            )
                            ingest([item], classifier, store, owner)
                            count += 1
                    with store.edit("social_cursor", state_id, owner) as saved:
                        saved["cursor"] = cursor
                    if count >= maximum:
                        return
        except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as exc:
            print(
                f"Disconnected ({type(exc).__name__}); retrying in {delay}s", flush=True
            )
            await asyncio.sleep(delay)
            delay = min(60, delay * 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True)
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "SPY"])
    parser.add_argument("--max-posts", type=int, default=100)
    parser.add_argument(
        "--classifier", choices=["baseline", "distilbert"], default="distilbert"
    )
    args = parser.parse_args()
    asyncio.run(main(args.owner, args.symbols, args.max_posts, args.classifier))
