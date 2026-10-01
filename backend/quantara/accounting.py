from copy import deepcopy

from .schemas import Portfolio, Transaction


def ledger(portfolio: Portfolio):
    accounts = {}
    seen = set()
    for tx in sorted(portfolio.transactions, key=lambda t: t.timestamp):
        if tx.external_id in seen:
            raise ValueError("Duplicate transaction external_id")
        seen.add(tx.external_id)
        if tx.affiliated:
            continue  # Supplied related-account activity informs tax checks, not owned assets.
        if tx.currency != portfolio.currency:
            raise ValueError(
                "Trading ledger requires transactions in the account currency"
            )
        account = accounts.setdefault(
            tx.account, {"cash": 0.0, "holdings": {}, "account_type": tx.account_type}
        )
        if account["account_type"] != tx.account_type:
            raise ValueError(
                "An account cannot mix taxable and registered classifications"
            )
        cash, positions = account["cash"], account["holdings"]
        symbol = tx.symbol
        qty = positions.get(symbol, 0.0)
        if tx.type == "deposit":
            cash += tx.amount
        elif tx.type == "withdrawal":
            cash -= tx.amount + tx.fee
        elif tx.type == "buy":
            cash -= tx.quantity * tx.price + tx.fee
            positions[symbol] = qty + tx.quantity
        elif tx.type == "sell":
            if tx.quantity > qty + 1e-8:
                raise ValueError(f"Insufficient {symbol} shares")
            cash += tx.quantity * tx.price - tx.fee
            positions[symbol] = max(0, qty - tx.quantity)
        elif tx.type == "split":
            positions[symbol] = qty * tx.quantity
        elif tx.type == "dividend":
            cash += tx.amount - tx.fee
        elif tx.type == "adjustment":
            pass  # Cost-basis adjustments are tax-only and never create trading cash.
        if cash < -1e-6:
            raise ValueError("Insufficient cash in account " + tx.account)
        account["cash"] = cash
    cash = sum(a["cash"] for a in accounts.values())
    positions = {}
    for account in accounts.values():
        for symbol, quantity in account["holdings"].items():
            positions[symbol] = positions.get(symbol, 0) + quantity
    return {
        "cash": round(cash, 8),
        "holdings": {s: q for s, q in positions.items() if q > 1e-8},
        "currency": portfolio.currency,
        "accounts": accounts,
    }


def import_transactions(portfolio, transactions):
    value = deepcopy(portfolio)
    existing = {t["external_id"]: t for t in value["transactions"]}
    added = 0
    for raw in transactions:
        tx = Transaction.model_validate(raw).model_dump(mode="json")
        if tx["external_id"] in existing:
            if tx != existing[tx["external_id"]]:
                raise ValueError(
                    "An existing external_id has different transaction contents"
                )
            continue
        value["transactions"].append(tx)
        existing[tx["external_id"]] = tx
        added += 1
    ledger(Portfolio.model_validate({k: v for k, v in value.items() if k != "id"}))
    return value, added
