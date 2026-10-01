from pathlib import Path
from uuid import uuid4
from hashlib import sha256
from importlib.metadata import version as package_version

import numpy as np

from .market import aligned_prices, validate_sessions, version
from .model_runtime import serialized_training
from .schemas import Backtest, Strategy
from .telemetry import publish
from .simulation import (
    ENGINE_VERSION,
    execute,
    groups,
    initial_state,
    run,
    submit_targets,
)

OBSERVATION_VERSION = 2


def observation(history, state, symbols):
    values = history[symbols].pct_change().fillna(0).to_numpy()[-10:]
    padded = np.zeros((10, len(symbols)))
    padded[-len(values) :] = np.clip(values, -1, 1)
    # Features use total returns, but shares and cash must use contemporaneous raw prices.
    last = state.get("marks", {})
    if set(symbols) - set(last):
        raise ValueError("RL observations require raw execution marks for every symbol")
    value = state["cash"] + sum(state["holdings"].get(s, 0) * last[s] for s in symbols)
    weights = [
        state["holdings"].get(s, 0) * last[s] / max(value, 1e-9) for s in symbols
    ]
    return np.r_[padded.ravel(), weights, state["cash"] / max(value, 1e-9)].astype(
        np.float32
    )


def allocation(action, symbols):
    values = np.asarray(action, dtype=float)
    if values.shape != (len(symbols),) or not np.isfinite(values).all():
        raise ValueError("Policy actions must contain one finite value per symbol")
    values = np.clip(values, 0, 1)
    # Sum below one leaves cash; sum above one becomes fully invested.
    values /= max(1, values.sum())
    return dict(zip(symbols, map(float, values)))


def make_environment(dataset, symbols, config):
    import gymnasium as gym

    validate_sessions(dataset)

    class TradingEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.action_space = gym.spaces.Box(
                0, 1, shape=(len(symbols),), dtype=np.float32
            )
            self.observation_space = gym.spaces.Box(
                -2, 2, shape=(11 * len(symbols) + 1,), dtype=np.float32
            )
            self.frame = aligned_prices(dataset, symbols)
            self.bars = groups(dataset)
            self.strategy = Strategy(
                name="RL environment", type="buy_hold", symbols=symbols
            )

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.state, self.cursor = initial_state(config.capital), 0
            self.ended = False
            ts, bars = self.bars[0]
            execute(self.state, bars, ts, config, dataset.actions)
            return observation(self.frame.loc[:ts], self.state, symbols), {}

        def step(self, action):
            if self.ended:
                raise ValueError("Episode has ended; reset before stepping again")
            ts, bars = self.bars[self.cursor]
            before = self.state["equity"][-1]["equity"]
            submit_targets(
                self.state, bars, allocation(action, symbols), self.strategy, ts
            )
            self.cursor += 1
            ts, bars = self.bars[self.cursor]
            execute(self.state, bars, ts, config, dataset.actions)
            after = self.state["equity"][-1]["equity"]
            reward = float(np.log(max(after, 1e-9) / max(before, 1e-9)))
            self.ended = self.cursor == len(self.bars) - 1
            return (
                observation(self.frame.loc[:ts], self.state, symbols),
                reward,
                self.ended,
                False,
                {},
            )

    return TradingEnv()


@serialized_training
def trained_policy(record, runtime):
    if record.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError(
            "This policy uses obsolete accounting observations; retrain the model"
        )
    path = Path(runtime) / "models" / (record["checkpoint"] + ".zip")
    if path.parent.resolve() != (Path(runtime) / "models").resolve():
        raise ValueError("Invalid checkpoint path")
    if not path.is_file():
        raise ValueError("Trained checkpoint is missing; retrain or restore the model")
    if sha256(path.read_bytes()).hexdigest() != record.get("checkpoint_sha256"):
        raise ValueError("Checkpoint checksum differs from the training record")
    from stable_baselines3 import DDPG, PPO

    model = (DDPG if record["algorithm"] == "DDPG" else PPO).load(path, device="cpu")
    symbols = record["symbols"]

    def policy(history, state):
        action, _ = model.predict(
            observation(history, state, symbols), deterministic=True
        )
        return allocation(action, symbols)

    return policy


@serialized_training
def train(request, dataset, runtime, progress=lambda *_: None):
    progress(0.02, "Preparing the training environment")
    from stable_baselines3 import DDPG, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed
    import torch

    validate_sessions(dataset)
    aligned_prices(dataset, list(dict.fromkeys(request.symbols + [request.benchmark])))
    torch.set_num_threads(1)
    set_random_seed(request.seed)
    sessions = sorted({b.timestamp.date() for b in dataset.bars})
    cut = max(1, int(len(sessions) * request.train_fraction))
    test_cut = int(
        len(sessions) * (request.train_fraction + request.validation_fraction)
    )
    if cut < 3 or test_cut <= cut or test_cut >= len(sessions) - 2:
        raise ValueError(
            "At least three sessions per training and test partition are required"
        )
    train_data = dataset.model_copy(
        update={
            "bars": [b for b in dataset.bars if b.timestamp.date() < sessions[cut]],
            "actions": [
                a for a in dataset.actions if a.timestamp.date() < sessions[cut]
            ],
        }
    )
    config = Backtest(
        dataset_id=request.dataset_id,
        strategy_id="training",
        train_fraction=request.train_fraction,
        validation_fraction=request.validation_fraction,
        capital=request.capital,
        commission=request.commission,
        slippage_bps=request.slippage_bps,
        spread_bps=request.spread_bps,
        participation=request.participation,
        benchmark=request.benchmark,
    )
    env = make_environment(train_data, request.symbols, config)
    training_history, episodes = [], []

    class Progress(BaseCallback):
        def __init__(self):
            super().__init__()
            self.window, self.episode_reward, self.episode_steps = [], 0.0, 0

        def snapshot(self):
            publish(progress, {"phase": "training", "steps": self.num_timesteps, "requested_steps": request.timesteps, "training_history": training_history[-200:], "episodes": episodes[-200:]})

        def _on_step(self):
            reward = float(self.locals["rewards"][0])
            self.window.append(reward)
            self.episode_reward += reward
            self.episode_steps += 1
            if bool(self.locals["dones"][0]):
                episodes.append({"step": self.num_timesteps, "episode_return": float(np.expm1(self.episode_reward)), "length": self.episode_steps})
                self.episode_reward, self.episode_steps = 0.0, 0
            if self.num_timesteps % 100 == 0:
                training_history.append({"step": self.num_timesteps, "mean_step_reward": float(np.mean(self.window)), "window_steps": len(self.window)})
                self.window = []
                progress(
                    min(0.8, 0.8 * self.num_timesteps / request.timesteps),
                    f"Training {self.num_timesteps} steps",
                )
                self.snapshot()
            return True

        def _on_training_end(self):
            if self.window:
                training_history.append({"step": self.num_timesteps, "mean_step_reward": float(np.mean(self.window)), "window_steps": len(self.window)})
            self.snapshot()

    algorithm = DDPG if request.algorithm == "DDPG" else PPO
    kwargs = (
        {"learning_starts": 100, "buffer_size": 100000}
        if request.algorithm == "DDPG"
        else {"n_steps": 128, "batch_size": 32}
    )
    model = algorithm(
        "MlpPolicy", env, seed=request.seed, verbose=0, device="cpu", **kwargs
    )
    model.learn(total_timesteps=request.timesteps, callback=Progress())
    if not model._n_updates:
        raise ValueError("Training did not perform a gradient update")
    directory = Path(runtime) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    identifier = str(uuid4())
    model.save(directory / identifier)
    parameters = sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        parameters.update(name.encode())
        parameters.update(tensor.detach().cpu().numpy().tobytes())
    record = {
        "name": f"{request.algorithm} seed {request.seed}",
        "algorithm": request.algorithm,
        "checkpoint": identifier,
        "symbols": request.symbols,
        "seed": request.seed,
        "timesteps": model.num_timesteps,
        "requested_timesteps": request.timesteps,
        "training_updates": model._n_updates,
        "training_history": training_history,
        "training_episodes": episodes,
        "training_metric_scope": "Mean log portfolio return per environment step, sampled in windows of up to 100 steps on training data only. Episode returns compound those same rewards. Training reward is not held-out performance.",
        "training_update_definition": "Stable-Baselines3 update counter: PPO optimization epochs; DDPG gradient steps",
        "parameter_sha256": parameters.hexdigest(),
        "checkpoint_sha256": sha256(
            (directory / (identifier + ".zip")).read_bytes()
        ).hexdigest(),
        "dataset_id": request.dataset_id,
        "dataset_version": version(dataset),
        "training_data_version": version(train_data),
        "train_end": str(sessions[cut - 1]),
        "validation_end": str(sessions[test_cut - 1]),
        "test_start": str(sessions[test_cut]),
        "observation_version": OBSERVATION_VERSION,
        "engine_version": ENGINE_VERSION,
        "runtime_versions": {
            name: package_version(name)
            for name in ("numpy", "torch", "gymnasium", "stable-baselines3")
        },
        "costs": config.model_dump(mode="json"),
    }
    policy = trained_policy(record, runtime)
    strategy = Strategy(
        name=record["name"], type="rl", symbols=request.symbols, model_id=identifier
    )
    evaluation_config = config.model_copy(update={"evaluation": "test"})
    progress(0.85, "Evaluating held-out data against simple strategies")
    evaluated = run(dataset, strategy, evaluation_config, policy=policy)
    comparisons = {}
    curves = {"policy": evaluated["state"]["equity"], "benchmark": evaluated["benchmark_curve"]}
    for template in ("buy_hold", "sma", "momentum"):
        simple = Strategy(name=template, type=template, symbols=request.symbols)
        comparison = run(dataset, simple, evaluation_config)
        comparisons[template] = comparison["metrics"]
        curves[template] = comparison["state"]["equity"]
    # Retain real sampled valuations for charts without duplicating enormous intraday audit logs.
    def sample_curve(curve):
        stride = max(1, int(np.ceil((len(curve) - 1) / 999)))
        sampled = curve[::stride]
        return sampled if sampled[-1] == curve[-1] else sampled + [curve[-1]]
    record["evaluation"] = {
        "policy": evaluated["metrics"],
        "comparisons": comparisons,
        "scope": "Single seeded run, untouched test partition; no outperformance guarantee",
        "period": evaluated["period"],
        "partitions": evaluated["partitions"],
        "execution_assumptions": evaluated["execution_assumptions"],
        "curves": {name: sample_curve(curve) for name, curve in curves.items()},
        "curve_scope": "Up to 1000 sampled actual valuations per series, including endpoints. Policy and templates share test dates and execution costs; the benchmark excludes simulated trading costs.",
        "benchmark": evaluated["benchmark_metrics"],
    }
    progress(1, "Training and held-out evaluation complete")
    return record
