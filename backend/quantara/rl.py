from pathlib import Path
from uuid import uuid4

import numpy as np

from .market import aligned_prices, version
from .schemas import Backtest, Strategy
from .simulation import execute, groups, initial_state, run, submit_targets


def observation(history, state, symbols):
    values = history[symbols].pct_change().fillna(0).to_numpy()[-10:]
    padded = np.zeros((10, len(symbols)))
    padded[-len(values) :] = np.clip(values, -1, 1)
    last = history.iloc[-1]
    value = state["cash"] + sum(state["holdings"].get(s, 0) * last[s] for s in symbols)
    weights = [
        state["holdings"].get(s, 0) * last[s] / max(value, 1e-9) for s in symbols
    ]
    return np.r_[padded.ravel(), weights, state["cash"] / max(value, 1e-9)].astype(
        np.float32
    )


def allocation(action, symbols):
    values = np.clip(np.asarray(action, dtype=float), 0, 1)
    # Sum below one leaves cash; sum above one becomes fully invested.
    values /= max(1, values.sum())
    return dict(zip(symbols, map(float, values)))


def make_environment(dataset, symbols, config):
    import gymnasium as gym

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
            ts, bars = self.bars[0]
            execute(self.state, bars, ts, config, dataset.actions)
            return observation(self.frame.loc[:ts], self.state, symbols), {}

        def step(self, action):
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
            return (
                observation(self.frame.loc[:ts], self.state, symbols),
                reward,
                self.cursor == len(self.bars) - 1,
                False,
                {},
            )

    return TradingEnv()


def trained_policy(record, runtime):
    from stable_baselines3 import DDPG, PPO

    path = Path(runtime) / "models" / (record["checkpoint"] + ".zip")
    if path.parent.resolve() != (Path(runtime) / "models").resolve():
        raise ValueError("Invalid checkpoint path")
    model = (DDPG if record["algorithm"] == "DDPG" else PPO).load(path, device="cpu")
    symbols = record["symbols"]

    def policy(history, state):
        action, _ = model.predict(
            observation(history, state, symbols), deterministic=True
        )
        return allocation(action, symbols)

    return policy


def train(request, dataset, runtime, progress=lambda *_: None):
    from stable_baselines3 import DDPG, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(request.seed)
    sessions = sorted({b.timestamp.date() for b in dataset.bars})
    cut = max(1, int(len(sessions) * request.train_fraction))
    test_cut = int(
        len(sessions) * (request.train_fraction + request.validation_fraction)
    )
    if cut < 3 or test_cut >= len(sessions) - 2:
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
    )
    env = make_environment(train_data, request.symbols, config)

    class Progress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 100 == 0:
                progress(
                    min(0.8, 0.8 * self.num_timesteps / request.timesteps),
                    f"Training {self.num_timesteps} steps",
                )
            return True

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
    directory = Path(runtime) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    identifier = str(uuid4())
    model.save(directory / identifier)
    record = {
        "name": f"{request.algorithm} seed {request.seed}",
        "algorithm": request.algorithm,
        "checkpoint": identifier,
        "symbols": request.symbols,
        "seed": request.seed,
        "timesteps": request.timesteps,
        "dataset_id": request.dataset_id,
        "dataset_version": version(dataset),
        "train_end": str(sessions[cut - 1]),
        "validation_end": str(sessions[test_cut - 1]),
        "test_start": str(sessions[test_cut]),
        "observation_version": 1,
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
    for template in ("buy_hold", "sma", "momentum"):
        simple = Strategy(name=template, type=template, symbols=request.symbols)
        comparisons[template] = run(dataset, simple, evaluation_config)["metrics"]
    record["evaluation"] = {
        "policy": evaluated["metrics"],
        "comparisons": comparisons,
        "scope": "Single seeded run, untouched test partition; no outperformance guarantee",
    }
    progress(1, "Training and held-out evaluation complete")
    return record
