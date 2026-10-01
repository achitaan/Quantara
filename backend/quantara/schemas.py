from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Number = Annotated[float, Field(allow_inf_nan=False)]
Positive = Annotated[Number, Field(gt=0)]
Nonnegative = Annotated[Number, Field(ge=0)]
Symbol = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9.\-]{0,14}$")]
Method = Literal[
    "equal_weight",
    "mean_variance",
    "min_volatility",
    "risk_parity",
    "black_litterman",
    "cvar",
]


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Login(Schema):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class Bar(Schema):
    timestamp: datetime
    symbol: Symbol
    open: Positive
    high: Positive
    low: Positive
    close: Positive
    volume: Nonnegative

    @field_validator("timestamp")
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None:
            raise ValueError("Bar timestamp must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def ohlc(self):
        if (
            not self.low
            <= min(self.open, self.close)
            <= max(self.open, self.close)
            <= self.high
        ):
            raise ValueError("Invalid OHLC bounds")
        return self


class CorporateAction(Schema):
    timestamp: datetime
    symbol: Symbol
    type: Literal["split", "dividend"]
    amount: Positive

    @field_validator("timestamp")
    @classmethod
    def aware(cls, value):
        return Bar.aware(value)


class Dataset(Schema):
    name: str = Field(min_length=1, max_length=120)
    interval: Literal["1d", "1m", "5m"] = "1d"
    source: str = Field(min_length=1, max_length=100)
    coverage: str = Field(min_length=1, max_length=500)
    fixture: bool = False
    bars: list[Bar] = Field(min_length=3, max_length=200000)
    actions: list[CorporateAction] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique(self):
        keys = [(b.timestamp, b.symbol) for b in self.bars]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate symbol/timestamp bars")
        self.bars.sort(key=lambda b: (b.timestamp, b.symbol))
        action_keys = [(a.timestamp, a.symbol, a.type) for a in self.actions]
        if len(set(action_keys)) != len(action_keys):
            raise ValueError("Duplicate or conflicting corporate actions")
        if set(a.symbol for a in self.actions) - set(b.symbol for b in self.bars):
            raise ValueError("Corporate-action symbols must have price history")
        self.actions.sort(key=lambda a: (a.timestamp, a.type != "split", a.symbol))
        return self


class MarketFetch(Schema):
    symbols: list[Symbol] = Field(min_length=1, max_length=30)
    start: datetime
    end: datetime
    interval: Literal["1d", "1m", "5m"] = "1d"
    provider: Literal["yahoo", "alpaca"] = "yahoo"

    @model_validator(mode="after")
    def range(self):
        if self.start >= self.end:
            raise ValueError("start must precede end")
        return self


class Transaction(Schema):
    external_id: str = Field(min_length=1, max_length=150)
    timestamp: datetime
    type: Literal[
        "deposit", "withdrawal", "buy", "sell", "dividend", "split", "adjustment"
    ]
    symbol: Symbol | None = None
    quantity: Nonnegative = 0
    price: Nonnegative = 0
    amount: Number = 0
    fee: Nonnegative = 0
    fx_cad: Positive | None = None
    currency: Literal["USD", "CAD"] = "USD"
    account: str = "main"
    account_type: Literal["taxable", "registered"] = "taxable"
    affiliated: bool = False

    @field_validator("timestamp")
    @classmethod
    def aware(cls, value):
        return Bar.aware(value)

    @model_validator(mode="after")
    def trading(self):
        if (
            self.type in ("buy", "sell", "split", "dividend", "adjustment")
            and not self.symbol
        ):
            raise ValueError("A symbol is required")
        if self.type in ("buy", "sell", "split") and self.quantity <= 0:
            raise ValueError("A positive quantity is required")
        if self.type in ("buy", "sell") and self.price <= 0:
            raise ValueError("A positive price is required")
        if self.type in ("deposit", "withdrawal") and self.amount <= 0:
            raise ValueError("A positive cash amount is required")
        return self


class Portfolio(Schema):
    name: str = Field(min_length=1, max_length=120)
    currency: Literal["USD", "CAD"] = "USD"
    transactions: list[Transaction] = Field(default_factory=list, max_length=20000)


class SettingsInput(Schema):
    watchlist: list[Symbol] = Field(default_factory=list, max_length=30)
    theme: Literal["light", "dark"] = "dark"


class Optimize(Schema):
    dataset_id: str
    method: Method = "equal_weight"
    holdings: dict[Symbol, Nonnegative] = Field(default_factory=dict)
    max_weight: Positive = Field(default=1, le=1)
    rf: Number = 0.02
    allow_short: bool = False
    views: dict[Symbol, Number] = Field(default_factory=dict)
    confidence: Positive = Field(default=0.5, lt=1)
    risk_aversion: Positive = 2.5


class Strategy(Schema):
    name: str = Field(min_length=1, max_length=120)
    type: Literal[
        "buy_hold", "sma", "rsi", "momentum", "rebalance", "sentiment", "rl"
    ] = "buy_hold"
    symbols: list[Symbol] = Field(min_length=1, max_length=30)
    fast: int = Field(default=10, ge=2, le=1000)
    slow: int = Field(default=30, ge=3, le=2000)
    rsi_buy: Number = Field(default=30, ge=0, le=100)
    rsi_sell: Number = Field(default=70, ge=0, le=100)
    rebalance_every: int = Field(default=20, ge=1, le=10000)
    method: Method = "equal_weight"
    limit_offset_bps: Nonnegative | None = Field(default=None, le=1000)
    sentiment_threshold: Number = Field(default=0.2, ge=-1, le=1)
    model_id: str | None = None

    @model_validator(mode="after")
    def rules(self):
        if self.fast >= self.slow or self.rsi_buy >= self.rsi_sell:
            raise ValueError("fast must be below slow and rsi_buy below rsi_sell")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Symbols must be unique")
        if self.type == "rebalance" and self.method == "black_litterman":
            raise ValueError(
                "Black-Litterman requires explicit views; use the optimization interface"
            )
        if self.type == "rl" and not self.model_id:
            raise ValueError("RL strategies require a model_id")
        return self


class Backtest(Schema):
    dataset_id: str
    strategy_id: str
    capital: Positive = 100000
    commission: Nonnegative = 0
    slippage_bps: Nonnegative = Field(default=5, le=1000)
    spread_bps: Nonnegative = Field(default=0, le=1000)
    participation: Positive = Field(default=0.01, le=1)
    benchmark: Symbol = "SPY"
    start: datetime | None = None
    end: datetime | None = None
    train_fraction: Positive = Field(default=0.6, lt=1)
    validation_fraction: Positive = Field(default=0.2, lt=1)
    evaluation: Literal["full", "test", "walk_forward"] = "full"

    @model_validator(mode="after")
    def splits(self):
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("A held-out test period is required")
        if self.start and self.end and self.start >= self.end:
            raise ValueError("start must precede end")
        for value in (self.start, self.end):
            if value and value.tzinfo is None:
                raise ValueError("Date boundaries need timezones")
        return self


class PaperCreate(Schema):
    name: str = Field(min_length=1, max_length=120)
    config: Backtest
    mode: Literal["replay", "forward"] = "replay"
    feed: Literal["alpaca", "yahoo"] = "alpaca"


class PaperControl(Schema):
    action: Literal["start", "pause", "resume", "stop"]


class RiskRequest(Schema):
    dataset_id: str
    weights: dict[Symbol, Nonnegative]
    benchmark: Symbol = "SPY"
    confidence: Positive = Field(default=0.95, lt=1)
    shocks: dict[Symbol, Number] = Field(default_factory=dict)


class CashTransaction(Schema):
    external_id: str
    timestamp: datetime
    amount: Number
    category: str = Field(min_length=1, max_length=100)
    currency: Literal["USD", "CAD"] = "CAD"

    @field_validator("timestamp")
    @classmethod
    def aware(cls, value):
        return Bar.aware(value)


class Cashflow(Schema):
    transactions: list[CashTransaction] = Field(min_length=2, max_length=20000)
    balance: Number = 10000
    currency: Literal["USD", "CAD"] = "CAD"
    days: int = Field(default=30, ge=1, le=365)
    method: Literal["baseline", "prophet", "lstm"] = "baseline"
    threshold: Number = 0


class TaxRequest(Schema):
    portfolio_id: str
    as_of: datetime
    affiliated_records_complete: bool = False
    marks_cad: dict[Symbol, Positive] = Field(default_factory=dict)

    @field_validator("as_of")
    @classmethod
    def aware(cls, value):
        return Bar.aware(value)


class NewsItem(Schema):
    external_id: str
    timestamp: datetime
    available_at: datetime
    symbols: list[Symbol] = Field(min_length=1, max_length=30)
    text: str = Field(min_length=1, max_length=5000)
    source: str
    url: str = ""
    fixture: bool = False

    @field_validator("timestamp", "available_at")
    @classmethod
    def aware(cls, value):
        return Bar.aware(value)

    @model_validator(mode="after")
    def availability(self):
        if self.available_at < self.timestamp:
            raise ValueError("available_at cannot precede publication")
        return self


class NewsImport(Schema):
    items: list[NewsItem] = Field(min_length=1, max_length=10000)
    method: Literal["baseline", "distilbert"] = "baseline"


class NewsImpact(Schema):
    dataset_id: str
    horizon_bars: int = Field(default=5, ge=1, le=1000)


class TrainRequest(Schema):
    dataset_id: str
    symbols: list[Symbol] = Field(min_length=1, max_length=30)
    algorithm: Literal["DDPG", "PPO"] = "PPO"
    timesteps: int = Field(default=2000, ge=100, le=1000000)
    seed: int = Field(default=42, ge=0)
    train_fraction: Positive = Field(default=0.6, lt=0.8)
    validation_fraction: Positive = Field(default=0.2, lt=0.4)
    capital: Positive = 100000
    commission: Nonnegative = 0
    slippage_bps: Nonnegative = Field(default=5, le=1000)
    spread_bps: Nonnegative = Field(default=0, le=1000)
    participation: Positive = Field(default=0.01, le=1)
    benchmark: Symbol = "SPY"

    @model_validator(mode="after")
    def split(self):
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("A test partition is required")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Symbols must be unique")
        if self.algorithm == "DDPG" and self.timesteps <= 100:
            raise ValueError("DDPG needs more than 100 steps to learn after warmup")
        return self


class Chat(Schema):
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None
    portfolio_id: str | None = None
    dataset_id: str | None = None
    use_rag: bool = True
    show_thinking: bool = False
    show_reflection: bool = False


class DocumentInput(Schema):
    name: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=1000000)
    page: int = Field(default=1, ge=1)


class PlaidExchange(Schema):
    public_token: str = Field(min_length=1, max_length=1000)


class PaperStep(Schema):
    bars: int = Field(default=1, ge=1, le=1000)


class TransactionsInput(Schema):
    transactions: list[Transaction] = Field(min_length=1, max_length=20000)


class DatasetView(Dataset):
    id: str


class StrategyView(Strategy):
    id: str


class Balance(Schema):
    cash: Number
    holdings: dict[str, Number]
    currency: str
    accounts: dict[str, dict]


class PortfolioView(Portfolio):
    id: str
    balance: Balance


class JobView(Schema):
    id: str
    operation: str
    arguments: dict
    status: Literal["queued", "running", "complete", "failed", "cancelled"]
    progress: Number
    message: str
    created_at: str
    updated_at: str
    result: dict | list | None = None
    error: str | None = None
    attempt: str | None = None


class MarketInfo(Schema):
    id: str
    name: str
    interval: str
    source: str
    coverage: str
    fixture: bool
    symbols: list[str]
    start: str
    end: str
    bars_count: int
    version: str


class SavedResult(Schema):
    id: str
    name: str
    result: dict


class NewsFetch(Schema):
    symbols: list[Symbol]
    start: datetime
    end: datetime
    method: Literal["baseline", "distilbert"] = "distilbert"
