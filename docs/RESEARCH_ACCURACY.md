# Simulation and training audit — 2026-10-01

The execution and training workflows have been audited against independent accounting examples, adversarial input changes, restart comparisons and real local model runs. This establishes tested mechanics, not prediction accuracy on unseen real data.

## Corrections

- RL return features retain point-in-time total returns, while portfolio weights now use raw contemporaneous share prices. Splits and cash dividends therefore reconcile correctly in observations and rewards.
- Paper replay honors the same date range and chronological evaluation partition as backtests. Serialized restarts reproduce the entire state exactly, including orders, fills, cash, shares and equity.
- Walk-forward fold metrics include the prior valuation, preserving returns at every boundary. Fold returns compound to the complete evaluation return. These are fixed-policy test windows, without per-fold retraining.
- Equal-weight rebalancing works on the first bar; it no longer unnecessarily invokes a statistical optimizer before enough history exists.
- Simulations reject missing execution sessions or benchmark bars. Dataset imports reject duplicate/conflicting corporate actions. A newly discovered action affecting already executed history requires account reconstruction, rather than silently altering current holdings.
- Sentiment decisions select the latest available article regardless of import order and exclude future articles.
- Neural training and model initialization share a process lock to prevent concurrent jobs from interfering with global random seeds. Seeded training is reproducible on the tested CPU/runtime; results are not promised bit-identical across hardware/library versions.
- Training saves requested and actual steps, update counts, data fingerprints, package versions, parameter/checkpoint checksums, costs and evaluation boundaries. DDPG requests that would finish before learning starts are rejected. Invalid actions and stepping after an episode ends are rejected.
- Backtests of trained policies disclose when evaluation dates include training/validation history and when execution settings differ from training. Training and paper-account controls expose commission, spread, slippage and volume participation.

## Evidence

The installed-runtime full suite passed **70 tests**, with one PostgreSQL test skipped in this run. Two additional tests also passed: a training/API integration test and a concurrent neural-job test. The first creates a real asynchronous training job, reloads the checkpoint, runs full and held-out simulations, verifies evaluation labels and cost changes, and checks that the held-out API metrics exactly match the training record. The second verifies identical parameter hashes and forecast results when PPO, DDPG and LSTM jobs run concurrently versus sequentially. Ruff, generated API types, production build and all three browser regressions pass.

The new accuracy suite reconstructs cash and holdings independently from corporate actions and fills for five strategy templates, with market and limit orders, partial fills, commissions, slippage and spread. It also checks known numerical risk values, closed-form allocations, one-minute/five-minute versus daily accounting, duplicate/late events, replay ranges, reward reconciliation and future-data isolation.

Changing all post-training price observations leaves PPO/DDPG parameter hashes unchanged for identical seeds. Changing forecast holdout values leaves baseline/Prophet/LSTM holdout predictions unchanged. The existing news-impact check confirms that altered test prices do not change fitted coefficients. Checkpoint corruption is detected before loading.

`runtime/research-audit.json` records the actual synthetic evaluations and associated checkpoint fingerprints. All policy comparisons use the same held-out dates and costs: USD 100,000 capital, USD 0.50 commission per fill, five basis points slippage, ten basis points spread and 1% volume participation. Each run requested 512 steps.

| Algorithm | Seed | Held-out return | Buy-and-hold return |
|---|---:|---:|---:|
| PPO | 42 | 0.80% | 1.76% |
| PPO | 71 | 0.57% | 1.76% |
| PPO | 99 | 0.21% | 1.76% |
| DDPG | 42 | 1.69% | 1.76% |
| DDPG | 71 | 0.79% | 1.76% |
| DDPG | 99 | 2.02% | 1.76% |

These small seeded synthetic runs verify executable learning and evaluation. One seed beating a comparator does not establish a useful trained policy. On a deterministic weekly cash-flow fixture, baseline/Prophet MAE was approximately zero and LSTM MAE was CAD 35.19; the neural model underperformed the simple baseline. The report preserves this outcome.

The local financial DistilBERT also correctly classified three illustrative positive, negative and neutral examples. These are model-wiring smoke checks, not a sentiment accuracy benchmark.

Fresh models and strategies are also registered in the running app: **Audited PPO research policy** and **Audited DDPG research policy**, each with a 2,000-step training target. Actual steps were 2,048 for PPO and 2,000 for DDPG. They use the default execution costs and have saved held-out simulations. The **Audited PPO held-out replay** account reproduced the complete corresponding backtest state exactly before being stopped. `runtime/research-app-acceptance.json` preserves this application workflow.

## Exact scope of the simulator

Signals use completed-bar history; an order can fill only on a later bar. Market fills use that bar's open plus modeled costs. Limit orders require a trade-through after costs; a touch alone does not fill. Volume participation constrains partial fills, and commissions apply to each fill. Sales fund purchases without borrowing. Open positions are marked at the final close, without forced-liquidation fees.

This is a bar-based research fill model. OHLCV does not contain queue position, the sequence of intrabar trades, bid/ask quotes or order-book depth. Market impact and queue priority are not modeled. Cash dividends are credited on the supplied event date, normally ex-date, rather than pay-date; withholding taxes and settlement delays are not modeled. A benchmark total-return curve excludes simulated trading costs, while the simple-strategy comparators use the same costs as the policy.

Forecast intervals resample held-out residuals independently. That calibrates research uncertainty but assumes no serial dependence and does not validate future interval coverage. Real-market predictive validation and a live-feed/reconnect soak test remain separate acceptance work.

## Existing saved results

Earlier simulation records remain preserved and are marked for rerunning in the result view. Observation-version-1 checkpoints are rejected for further execution and must be retrained; silently reusing weights with changed observations would invalidate their meaning. Existing reports are not overwritten.

## Reproduction

```powershell
$env:QUANTARA_MODEL_TESTS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/audit_research.py --include-models
```

Optional model dependencies and their local weights are required. The audit command writes isolated artifacts under `runtime/`; it does not change team accounts or saved records. Its default three seeds and step target are configurable. CI now runs the adversarial accuracy suite with the installed model runtime.

Stable-Baselines3 documents training-step targets as a lower bound; actual step counts are therefore recorded separately ([official API](https://stable-baselines3.readthedocs.io/en/master/modules/base.html)). Gymnasium's reset/step contract and environment checkers are used for episode and observation validation ([official API](https://gymnasium.farama.org/api/env/)).
