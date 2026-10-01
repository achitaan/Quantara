from collections import defaultdict

import numpy as np
import pandas as pd

from .model_runtime import serialized_training


def daily_transactions(request):
    ids = set()
    totals = defaultdict(float)
    for t in request.transactions:
        if t.external_id in ids:
            raise ValueError("Duplicate cash transaction ID")
        if t.currency != request.currency:
            raise ValueError(
                "Convert cash transactions into the forecast currency first"
            )
        ids.add(t.external_id)
        totals[pd.Timestamp(t.timestamp).normalize()] += t.amount
    series = pd.Series(totals).sort_index()
    return series.reindex(
        pd.date_range(series.index[0], series.index[-1], freq="1D"), fill_value=0
    )


def baseline(train, days):
    index = pd.date_range(
        train.index[-1] + pd.Timedelta(days=1), periods=days, freq="1D"
    )
    means = train.groupby(train.index.dayofweek).mean()
    return pd.Series(
        [means.get(ts.dayofweek, train.mean()) for ts in index], index=index
    )


def predict(train, days, method):
    if method == "baseline":
        return baseline(train, days), None
    return predict_model(train, days, method)


@serialized_training
def predict_model(train, days, method):
    if len(train) < 90:
        raise ValueError(
            "Prophet/LSTM require at least 90 days of chronological history"
        )
    if method == "prophet":
        from prophet import Prophet

        model = Prophet(daily_seasonality=False, yearly_seasonality=False)
        model.fit(
            pd.DataFrame({"ds": train.index.tz_localize(None), "y": train.values}),
            seed=42,
        )
        future = model.make_future_dataframe(periods=days, include_history=False)
        out = model.predict(future)
        index = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=days)
        return pd.Series(out.yhat.values, index=index), {
            "engine": "Prophet",
            "training_rows": len(train),
        }
    if method == "lstm":
        import torch

        torch.manual_seed(42)
        torch.set_num_threads(1)
        mean, scale = float(train.mean()), max(float(train.std()), 1)
        values = (train.values - mean) / scale
        window = 14
        x = torch.tensor(
            np.array([values[i - window : i] for i in range(window, len(values))]),
            dtype=torch.float32,
        ).unsqueeze(-1)
        y = torch.tensor(values[window:], dtype=torch.float32).unsqueeze(-1)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(1, 16, batch_first=True)
                self.output = torch.nn.Linear(16, 1)

            def forward(self, data):
                hidden, _ = self.lstm(data)
                return self.output(hidden[:, -1])

        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(60):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
        tail, forecast = list(values[-window:]), []
        model.eval()
        with torch.no_grad():
            for _ in range(days):
                result = model(
                    torch.tensor(tail[-window:], dtype=torch.float32).reshape(
                        1, window, 1
                    )
                ).item()
                tail.append(result)
                forecast.append(result * scale + mean)
        index = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=days)
        return pd.Series(forecast, index=index), {
            "engine": "torch LSTM",
            "seed": 42,
            "epochs": 60,
            "training_rows": len(train),
        }
    raise ValueError("Unknown forecast method")


def forecast(request, progress=lambda *_: None):
    series = daily_transactions(request)
    if len(series) < 14:
        raise ValueError("At least 14 calendar days are required")
    split = int(len(series) * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    progress(0.1, "Evaluating chronological holdout")
    validation, _ = predict(train, len(test), request.method)
    residuals = test.to_numpy() - validation.to_numpy()
    base = baseline(train, len(test))
    evaluation = {
        "train_end": train.index[-1].isoformat(),
        "test_start": test.index[0].isoformat(),
        "test_rows": len(test),
        "mae": float(np.abs(residuals).mean()),
        "baseline_mae": float(np.abs(test.values - base.values).mean()),
        "interval_method": "Empirical held-out residual bootstrap; research uncertainty",
        "interval_scope": "Holdout residuals also calibrate intervals; independent resampling assumes no serial dependence. Future coverage has not been validated.",
    }
    progress(0.5, "Refitting on recorded history")
    future, metadata = predict(series, request.days, request.method)
    rng = np.random.default_rng(42)
    paths = request.balance + np.cumsum(
        future.values[None, :] + rng.choice(residuals, (500, request.days)), axis=1
    )
    low, high = np.quantile(paths, [0.05, 0.95], axis=0)
    balance = request.balance
    rows = []
    for i, (ts, amount) in enumerate(future.items()):
        balance += float(amount)
        rows.append(
            {
                "date": ts.date().isoformat(),
                "net_flow": float(amount),
                "balance": balance,
                "lower": float(low[i]),
                "upper": float(high[i]),
                "shortfall": low[i] < request.threshold,
            }
        )
    categories = defaultdict(list)
    for t in request.transactions:
        categories[t.category].append(t)
    recurring = []
    for category, tx in categories.items():
        tx.sort(key=lambda t: t.timestamp)
        if len(tx) < 3:
            continue
        gaps = np.array([(b.timestamp - a.timestamp).days for a, b in zip(tx, tx[1:])])
        amounts = np.array([t.amount for t in tx])
        period = float(np.median(gaps))
        if (
            period > 0
            and np.median(np.abs(gaps - period)) <= 3
            and np.std(amounts) < max(1, abs(amounts.mean()) * 0.2)
        ):
            recurring.append(
                {
                    "category": category,
                    "period_days": period,
                    "amount": float(np.median(amounts)),
                    "next_date": (
                        pd.Timestamp(tx[-1].timestamp) + pd.Timedelta(days=period)
                    )
                    .date()
                    .isoformat(),
                }
            )
    progress(1, "Forecast complete")
    return {
        "method": request.method,
        "currency": request.currency,
        "forecast": rows,
        "recurring": recurring,
        "evaluation": evaluation,
        "model": metadata,
        "shortfall_alerts": [r["date"] for r in rows if r["shortfall"]],
    }
