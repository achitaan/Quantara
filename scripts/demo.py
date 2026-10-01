"""Automated, offline-first demonstration; saves every step and the final transcript."""

import argparse
from getpass import getpass
import os

from dotenv import load_dotenv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import httpx


def run(url, username, password, include_models=False):
    results = {}
    with httpx.Client(base_url=url, timeout=30) as client:

        def call(path, body=None):
            response = (
                client.post("/api/v1/" + path, json=body)
                if body is not None
                else client.get("/api/v1/" + path)
            )
            response.raise_for_status()
            return response.json()

        def job(path, body):
            value = call(path, body)
            for _ in range(1680):
                saved = call("jobs/" + value["id"])
                if saved["status"] == "complete":
                    return saved["result"]
                if saved["status"] in ("failed", "cancelled"):
                    raise ValueError(saved["error"] or saved["status"])
                time.sleep(0.25)
            raise TimeoutError("Demo job exceeded 420 seconds")

        call("auth/login", {"username": username, "password": password})
        d = call("demo", {})
        results["import"] = d
        results["risk"] = job(
            "risk",
            {"dataset_id": d["dataset_id"], "weights": {"AAPL": 0.5, "MSFT": 0.5}},
        )
        results["optimization"] = job(
            "portfolios/optimize",
            {
                "dataset_id": d["dataset_id"],
                "method": "risk_parity",
                "holdings": {"AAPL": 0.5, "MSFT": 0.5},
            },
        )
        config = {"dataset_id": d["dataset_id"], "strategy_id": d["strategy_id"]}
        results["backtest"] = job("backtests", config)
        paper = call(
            "paper-accounts", {"name": "Automated demo paper", "config": config}
        )
        call("paper-accounts/" + paper["id"] + "/control", {"action": "start"})
        results["paper"] = job("paper-accounts/" + paper["id"] + "/step", {"bars": 20})
        tx = call("cashflow/demo", {})["transactions"]
        results["forecast"] = job("cashflow", {"transactions": tx, "currency": "CAD"})
        bars = call("market/" + d["dataset_id"])["bars"]
        samples = [b for b in bars if b["symbol"] == "AAPL"][::10]
        items = [
            {
                "external_id": "demo-news-" + str(i),
                "timestamp": b["timestamp"],
                "available_at": b["timestamp"],
                "symbols": ["AAPL"],
                "text": "Strong profit growth" if i % 2 else "Weak profit decline",
                "source": "demo-fixture",
                "fixture": True,
            }
            for i, b in enumerate(samples)
        ]
        results["signals"] = job(
            "signals/import", {"items": items, "method": "baseline"}
        )
        results["news_evaluation"] = job(
            "signals/evaluate", {"dataset_id": d["dataset_id"], "horizon_bars": 5}
        )
        results["tax"] = job(
            "tax/research",
            {
                "portfolio_id": d["portfolio_id"],
                "as_of": datetime.now(timezone.utc).isoformat(),
                "marks_cad": {"AAPL": 150, "MSFT": 450},
            },
        )
        if include_models:
            results["neural_forecast"] = job(
                "cashflow",
                {
                    "transactions": tx,
                    "currency": "CAD",
                    "method": "lstm",
                    "days": 7,
                },
            )
            results["prophet_forecast"] = job(
                "cashflow",
                {
                    "transactions": tx,
                    "currency": "CAD",
                    "method": "prophet",
                    "days": 7,
                },
            )
            results["rl_models"] = []
            for algorithm in ("PPO", "DDPG"):
                trained = job(
                    "models/train",
                    {
                        "dataset_id": d["dataset_id"],
                        "symbols": ["AAPL", "MSFT"],
                        "algorithm": algorithm,
                        "timesteps": 128,
                        "seed": 42,
                    },
                )
                results["rl_models"].append(trained)
                strategy = call(
                    "strategies",
                    {
                        "name": algorithm + " demo policy",
                        "type": "rl",
                        "symbols": ["AAPL", "MSFT"],
                        "model_id": trained["id"],
                    },
                )
                results[algorithm + "_backtest"] = job(
                    "backtests",
                    {
                        "dataset_id": d["dataset_id"],
                        "strategy_id": strategy["id"],
                        "evaluation": "test",
                    },
                )
        results["explanation"] = job(
            "conversations/chat",
            {
                "message": "Explain the recorded risk, allocation and simulation results. Cite the records and identify fixture limitations.",
                "portfolio_id": d["portfolio_id"],
                "dataset_id": d["dataset_id"],
                "use_rag": False,
            },
        )
        results["health"] = call("health")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser.add_argument("--username", default=os.getenv("TEAM_USERNAME", "demo"))
    parser.add_argument("--output", default="runtime/demo-report.json")
    parser.add_argument(
        "--include-models",
        action="store_true",
        help="Run installed local forecasts and brief RL training; no performance claim",
    )
    args = parser.parse_args()
    result = run(
        args.url,
        args.username,
        os.getenv("TEAM_PASSWORD") or getpass("Password: "),
        args.include_models,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(target.resolve()),
                "fixture": True,
                "ai_mode": result["explanation"]["mode"],
                "ai_warning": result["explanation"]["warning"],
            },
            indent=2,
        )
    )
