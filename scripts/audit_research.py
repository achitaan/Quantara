"""Record reproducible simulation and actual model evaluations without changing team data."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from quantara import forecast, market, news, rl, simulation
from quantara.config import settings
from quantara.schemas import Backtest, Cashflow, Strategy, TrainRequest
from quantara.serialization import clean


def audit(include_models, timesteps, seeds, directory):
    dataset = market.fixture()
    config = Backtest(
        dataset_id="audit:seed42", strategy_id="audit", commission=0.5, spread_bps=10
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": market.version(dataset),
        "fixture": True,
        "scope": "Seeded synthetic workflow verification. These measured returns and errors do not establish performance on real market or transaction data.",
        "simulations": {},
        "training": [],
        "forecasts": {},
    }
    for template in ("buy_hold", "sma", "rsi", "momentum", "rebalance"):
        strategy = Strategy(name=template, type=template, symbols=["AAPL", "MSFT"])
        result = simulation.run(dataset, strategy, config)
        account = {
            "status": "running",
            "mode": "replay",
            "config": config.model_dump(mode="json"),
            "state": simulation.initial_state(config.capital),
        }
        while not account.get("replay_complete"):
            account = simulation.replay_step(
                json.loads(json.dumps(account)), dataset, strategy, 100
            )
        if account["state"] != result["state"]:
            raise AssertionError(f"Replay mismatch for {template}")
        report["simulations"][template] = {
            "metrics": result["metrics"],
            "fills": len(result["state"]["fills"]),
            "replay_matches": True,
            "period": result["period"],
        }
    for interval in ("1m", "5m"):
        data = market.fixture(interval)
        result = simulation.run(
            data, Strategy(name="Intraday hold", symbols=["AAPL", "MSFT"]), config
        )
        report["simulations"][interval] = {
            "metrics": result["metrics"],
            "fills": len(result["state"]["fills"]),
            "dataset_version": market.version(data),
            "period": result["period"],
        }
    report["execution_assumptions"] = simulation.EXECUTION_ASSUMPTIONS
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    tx = [
        dict(
            external_id=str(i),
            timestamp=start + timedelta(days=i),
            amount=100 if i % 7 == 0 else -10,
            category="weekly",
            currency="CAD",
        )
        for i in range(140)
    ]
    for method in ("baseline", "prophet", "lstm") if include_models else ("baseline",):
        result = forecast.forecast(Cashflow(transactions=tx, method=method, days=7))
        report["forecasts"][method] = {
            "evaluation": result["evaluation"],
            "model": result["model"],
            "fixture": True,
        }
    if include_models:
        examples = [
            (
                "The company reported strong profit growth and record revenue.",
                "positive",
            ),
            (
                "The company reported a large loss and cut its earnings forecast.",
                "negative",
            ),
            ("The company will publish its quarterly report on Tuesday.", "neutral"),
        ]
        report["sentiment_smoke_checks"] = []
        for text, expected in examples:
            actual = news.classify(text, "distilbert")
            if actual["label"] != expected:
                raise AssertionError(f"Unexpected sentiment label: {actual}")
            report["sentiment_smoke_checks"].append(
                {"text": text, "expected": expected, **actual}
            )
        for algorithm in ("PPO", "DDPG"):
            for seed in seeds:
                request = TrainRequest(
                    dataset_id="audit:seed42",
                    symbols=["AAPL", "MSFT"],
                    algorithm=algorithm,
                    timesteps=timesteps,
                    seed=seed,
                    commission=config.commission,
                    spread_bps=config.spread_bps,
                )
                print(
                    f"Training {algorithm}, seed {seed}, target {timesteps} steps",
                    flush=True,
                )
                record = rl.train(request, dataset, directory)
                report["training"].append(record)
    return clean(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-models", action="store_true")
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--seeds", default="42,71,99")
    parser.add_argument(
        "--output", type=Path, default=settings.runtime / "research-audit.json"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = audit(
        args.include_models,
        args.timesteps,
        [int(s) for s in args.seeds.split(",")],
        output.parent / "research-audit-models",
    )
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)
