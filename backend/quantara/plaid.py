"""Optional Plaid Sandbox only. Tokens never appear in browser responses."""

from datetime import datetime, timezone
import os

import httpx


def request(endpoint, body):
    client, secret = os.getenv("PLAID_CLIENT_ID"), os.getenv("PLAID_SECRET")
    if not client or not secret:
        raise ValueError("Configure PLAID_CLIENT_ID and PLAID_SECRET for Plaid Sandbox")
    response = httpx.post(
        "https://sandbox.plaid.com/" + endpoint,
        json={"client_id": client, "secret": secret, **body},
        timeout=30,
    )
    if response.status_code >= 400:
        raise ValueError(
            "Plaid Sandbox request failed: "
            + response.json().get("error_code", "unknown")
        )
    return response.json()


def link_token(owner):
    return request(
        "link/token/create",
        {
            "user": {"client_user_id": owner},
            "client_name": "Quantara Research",
            "products": ["transactions"],
            "country_codes": ["US", "CA"],
            "language": "en",
        },
    )["link_token"]


def exchange(public_token, store, owner):
    result = request("item/public_token/exchange", {"public_token": public_token})
    item = result["item_id"]
    try:
        store.get("plaid", "plaid:" + item, owner)
    except KeyError:
        store.create(
            "plaid",
            owner,
            {"access_token": result["access_token"], "cursor": ""},
            "plaid:" + item,
        )
    return {"item_id": item, "environment": "sandbox"}


def sync(item_id, store, owner):
    value = store.get("plaid", "plaid:" + item_id, owner)
    cursor, added, modified, removed = value["cursor"], [], [], []
    for _ in range(50):
        response = request(
            "transactions/sync",
            {"access_token": value["access_token"], "cursor": cursor, "count": 500},
        )
        added += response["added"]
        modified += response["modified"]
        removed += response["removed"]
        cursor = response["next_cursor"]
        if not response["has_more"]:
            break
    else:
        raise ValueError(
            "Plaid history exceeds sync limit; retry with a smaller Sandbox history"
        )
    try:
        book = store.get("cashbook", "cashbook:" + owner, owner)
    except KeyError:
        book = store.create(
            "cashbook", owner, {"transactions": []}, "cashbook:" + owner
        )
    with store.edit("cashbook", book["id"], owner) as saved:
        records = {t["external_id"]: t for t in saved["transactions"]}
        for t in added + modified:
            if t["pending"]:
                continue
            currency = t.get("iso_currency_code")
            if currency not in ("CAD", "USD"):
                raise ValueError("Sandbox transaction has an unsupported currency")
            key = "plaid:" + t["transaction_id"]
            records[key] = {
                "external_id": key,
                "timestamp": datetime.fromisoformat(t["date"])
                .replace(tzinfo=timezone.utc)
                .isoformat(),
                "amount": -t["amount"],
                "category": (t.get("personal_finance_category") or {}).get(
                    "primary", "uncategorized"
                ),
                "currency": currency,
            }
        for t in removed:
            records.pop("plaid:" + t["transaction_id"], None)
        saved["transactions"] = sorted(records.values(), key=lambda t: t["timestamp"])
    with store.edit("plaid", "plaid:" + item_id, owner) as saved:
        saved["cursor"] = cursor
    return {
        "added": len(added),
        "modified": len(modified),
        "removed": len(removed),
        "environment": "sandbox",
    }
