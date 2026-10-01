"""Canadian research ledger: pooled CAD ACB and conservative superficial-loss flags."""

from collections import defaultdict
from datetime import datetime, timedelta

from .schemas import Portfolio

CRA_LOSS = "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/personal-income/line-12700-capital-gains/capital-losses-deductions.html/1000"
CRA_FX = "https://www.canada.ca/en/revenue-agency/services/tax/individuals/topics/about-your-tax-return/tax-return/completing-a-tax-return/personal-income/line-12700-capital-gains/you-calculate-your-capital-gain-loss.html"


def superficial(sale, records, as_of, complete):
    low, high = sale.timestamp - timedelta(days=30), sale.timestamp + timedelta(days=30)
    related = [
        t
        for t in records
        if t.symbol == sale.symbol and t.timestamp <= min(high, as_of)
    ]
    # Normalize all quantities into the sale-date share units for splits within the window.
    splits = [t for t in records if t.symbol == sale.symbol and t.type == "split"]

    def units(t):
        q = t.quantity
        for split in splits:
            if t.timestamp < split.timestamp <= sale.timestamp:
                q *= split.quantity
            elif sale.timestamp < split.timestamp <= t.timestamp:
                q /= split.quantity
        return q

    purchases = [t for t in related if t.type == "buy" and low <= t.timestamp <= high]
    bought = sum(units(t) for t in purchases)
    owned = defaultdict(float)
    for t in related:
        if t.type in ("buy", "sell"):
            owned[(t.account, t.affiliated, t.account_type)] += units(t) * (
                1 if t.type == "buy" else -1
            )
    remaining = sum(max(0, q) for q in owned.values())
    denied_shares = min(sale.quantity, bought, remaining)
    replacement = [
        {
            "account": t.account,
            "affiliated": t.affiliated,
            "account_type": t.account_type,
            "quantity": units(t),
            "external_id": t.external_id,
        }
        for t in purchases
    ]
    status = "provisional" if high > as_of or not complete else "final"
    # Overlapping loss windows require individual replacement-lot allocation.
    overlap = any(
        t.type == "sell"
        and t.external_id != sale.external_id
        and low <= t.timestamp <= high
        for t in related
    )
    return {
        "denied_fraction": denied_shares / sale.quantity,
        "denied_shares": denied_shares,
        "window_start": low.isoformat(),
        "window_end": high.isoformat(),
        "status": status,
        "replacement_activity": replacement,
        "overlapping_disposals": overlap,
        "records_complete": complete,
    }


def research(portfolio, as_of, complete=False, marks=None):
    portfolio = Portfolio.model_validate(portfolio)
    records = sorted(portfolio.transactions, key=lambda t: (t.timestamp, t.external_id))
    pools = defaultdict(lambda: {"quantity": 0.0, "acb_cad": 0.0})
    dispositions, pending, incomplete = [], [], []
    for t in records:
        if t.timestamp > as_of:
            continue
        for adjustment in list(pending):
            if adjustment["effective"] <= t.timestamp:
                pools[adjustment["symbol"]]["acb_cad"] += adjustment["amount"]
                pending.remove(adjustment)
        if t.affiliated or t.account_type == "registered" or not t.symbol:
            continue
        pool = pools[t.symbol]
        if t.type in ("buy", "sell") and t.currency == "USD" and t.fx_cad is None:
            incomplete.append(
                {
                    "external_id": t.external_id,
                    "reason": "Missing transaction-date USD/CAD rate",
                }
            )
            continue
        fx = 1 if t.currency == "CAD" else t.fx_cad
        if t.type == "buy":
            pool["quantity"] += t.quantity
            pool["acb_cad"] += (t.quantity * t.price + t.fee) * fx
        elif t.type == "split":
            pool["quantity"] *= t.quantity
        elif t.type == "adjustment":
            pool["acb_cad"] += t.amount * (fx or 1)
        elif t.type == "sell":
            if t.quantity > pool["quantity"] + 1e-8:
                incomplete.append(
                    {
                        "external_id": t.external_id,
                        "reason": "Sale exceeds recorded taxable shares",
                    }
                )
                continue
            cost = pool["acb_cad"] * t.quantity / pool["quantity"]
            proceeds = (t.quantity * t.price - t.fee) * fx
            gain = proceeds - cost
            pool["quantity"] -= t.quantity
            pool["acb_cad"] -= cost
            row = {
                "external_id": t.external_id,
                "timestamp": t.timestamp.isoformat(),
                "symbol": t.symbol,
                "quantity": t.quantity,
                "proceeds_cad": proceeds,
                "allocated_acb_cad": cost,
                "gain_cad": gain,
            }
            if gain < 0:
                check = superficial(t, records, as_of, complete)
                denied = -gain * check["denied_fraction"]
                check["denied_loss_cad"] = denied
                row.update(
                    {"superficial_loss": check, "allowable_loss_cad": -gain - denied}
                )
                replacements = check["replacement_activity"]
                own_taxable = replacements and all(
                    not r["affiliated"] and r["account_type"] == "taxable"
                    for r in replacements
                )
                if (
                    denied
                    and check["status"] == "final"
                    and own_taxable
                    and not check["overlapping_disposals"]
                ):
                    pending.append(
                        {
                            "symbol": t.symbol,
                            "amount": denied,
                            "effective": t.timestamp
                            + timedelta(days=30, microseconds=1),
                        }
                    )
                    check["acb_treatment"] = (
                        "Added to own taxable pool after completed window"
                    )
                elif denied:
                    check["acb_treatment"] = (
                        "Registered replacements: denied loss is not added to taxable ACB"
                        if replacements
                        and all(r["account_type"] == "registered" for r in replacements)
                        else "Provisional or affiliated/overlapping replacements: manual ACB allocation required"
                    )
            dispositions.append(row)
    for adjustment in pending:
        if adjustment["effective"] <= as_of:
            pools[adjustment["symbol"]]["acb_cad"] += adjustment["amount"]
    proposals = []
    for symbol, pool in pools.items():
        if marks and symbol in marks and pool["quantity"] > 0:
            price_cad = marks[symbol]
            loss = pool["acb_cad"] - pool["quantity"] * price_cad
            if loss > 0:
                proposals.append(
                    {
                        "symbol": symbol,
                        "quantity": pool["quantity"],
                        "estimated_loss_cad": loss,
                        "status": "Research proposal; check ±30-day affiliated activity before selling",
                    }
                )
    return {
        "currency": "CAD",
        "as_of": as_of.isoformat(),
        "pools": dict(pools),
        "dispositions": dispositions,
        "incomplete_records": incomplete,
        "provisional": bool(incomplete)
        or not complete
        or any(
            d.get("superficial_loss", {}).get("status") == "provisional"
            for d in dispositions
        ),
        "harvesting_proposals": proposals,
        "harvesting_basis": (
            "CAD marks were supplied; proposals include only recorded taxable pools with an unrealized loss."
            if marks
            else "No CAD marks were supplied; harvesting valuation was not requested. Affiliated-record completeness does not prevent proposal generation."
        ),
        "record_completeness": {
            "affiliated_records_complete": complete,
            "own_record_issues": len(incomplete),
            "future_windows_pending": sum(
                datetime.fromisoformat(d["superficial_loss"]["window_end"]) > as_of
                for d in dispositions
                if "superficial_loss" in d
            ),
            "definition": "Supplied affiliated trades are checked even when completeness is false; false means additional activity may be missing and results remain provisional.",
        },
        "sources": [CRA_LOSS, CRA_FX],
        "scope": "Taxable pools across supplied own accounts; registered trades only inform replacement checks. "
        "Complex overlapping/affiliated ACB allocation is flagged for review. No tax-filing calculation.",
    }
