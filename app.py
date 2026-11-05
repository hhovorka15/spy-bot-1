"""SPY autonomous trading brain (Alpaca).

Single-process design: one background scan loop plus a Flask control surface
(run with a single Gunicorn worker). Webhook entries (BUY_LONG / SELL_SHORT) require
WEBHOOK_SECRET. Entries from the built-in signal engine are opt-in via AUTO_ENTRY=true.
"""

from __future__ import annotations

import atexit
import hmac
import json
import logging
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any, Callable, Final, NamedTuple, TypeVar
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
)
from flask import Flask, jsonify, request
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout
from urllib3.exceptions import ProtocolError

try:
    import fcntl
except ImportError:  # non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("spy-brain")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, os.environ.get("LOG_LEVEL", "INFO").strip().upper(), logging.INFO))
    logger.propagate = False
    return logger


log = _build_logger()


def log_event(level: int, event: str, **fields: Any) -> None:
    log.log(level, event, extra={"fields": fields})


_throttle_lock = threading.Lock()
_throttle_last: dict[str, float] = {}


def throttled(key: str, interval: float = 300.0) -> bool:
    """Return True when an event keyed by `key` may be emitted (rate-limited)."""
    now = time.monotonic()
    with _throttle_lock:
        last = _throttle_last.get(key)
        if last is not None and now - last < interval:
            return False
        if len(_throttle_last) > 256:
            _throttle_last.clear()
        _throttle_last[key] = now
    return True


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when an environment variable holds an invalid value."""


_TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES: Final = frozenset({"0", "false", "no", "off"})


def _env_raw(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_str(name: str, default: str) -> str:
    return _env_raw(name) or default


def _env_choice(name: str, default: str, choices: frozenset[str]) -> str:
    value = _env_str(name, default).upper()
    if value not in choices:
        raise ConfigError(f"{name} must be one of {sorted(choices)}, got {value!r}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_raw(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ConfigError(f"{name} must be a boolean, got {raw!r}")


def _check_range(name: str, value: float, minimum: float | None, maximum: float | None) -> None:
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}, got {value!r}")


def _env_int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = _env_raw(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    _check_range(name, value, minimum, maximum)
    return value


def _env_float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = _env_raw(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    _check_range(name, value, minimum, maximum)
    return value


@dataclass(frozen=True)
class Config:
    symbol: str
    check_seconds: int
    cooldown_minutes: int
    max_hold_minutes: int
    max_bar_age_seconds: int
    risk_mode: str
    risk_usd_per_trade: float
    risk_pct_per_trade: float
    max_position_pct_equity: float
    fallback_qty: int
    adx_trend_min: float
    tp_mult: float
    sl_mult: float
    ratchet_trig: float
    ratchet_lock: float
    use_1h_gate: bool
    use_15m_gate: bool
    use_vol_gate: bool
    max_daily_loss_usd: float
    max_trades_per_day: int
    preferred_feed: str
    auto_entry: bool
    paper: bool
    http_timeout: float
    webhook_secret: str
    admin_token: str
    kill_switch: bool
    autostart_worker: bool
    instance_lock_path: str

    @classmethod
    def from_env(cls) -> Config:
        symbol = _env_str("TRADE_SYMBOL", "SPY").upper()
        if not re.fullmatch(r"[A-Z][A-Z.\-]{0,9}", symbol):
            raise ConfigError(f"TRADE_SYMBOL is invalid: {symbol!r}")
        return cls(
            symbol=symbol,
            check_seconds=_env_int("CHECK_SECONDS", 30, 5, 3600),
            cooldown_minutes=_env_int("COOLDOWN_MINUTES", 15, 0, 1440),
            max_hold_minutes=_env_int("MAX_HOLD_MINUTES", 90, 1, 1440),
            max_bar_age_seconds=_env_int("MAX_BAR_AGE_SECONDS", 360, 60, 3600),
            risk_mode=_env_choice("RISK_MODE", "FIXED_USD", frozenset({"FIXED_USD", "PCT_EQUITY"})),
            risk_usd_per_trade=_env_float("RISK_USD_PER_TRADE", 50.0, 0.01),
            risk_pct_per_trade=_env_float("RISK_PCT_PER_TRADE", 0.50, 0.001, 100.0),
            max_position_pct_equity=_env_float("MAX_POSITION_PCT_EQUITY", 95.0, 0.1, 100.0),
            fallback_qty=_env_int("TRADE_QTY", 10, 1),
            adx_trend_min=_env_float("ADX_TREND_MIN", 25.0, 0.0, 100.0),
            tp_mult=_env_float("TB_TP_MULT", 2.25, 0.1, 50.0),
            sl_mult=_env_float("TB_SL_MULT", 1.25, 0.1, 50.0),
            ratchet_trig=_env_float("RATCHET_TRIG", 0.75, 0.05, 1.0),
            ratchet_lock=_env_float("RATCHET_LOCK", 0.35, 0.0, 50.0),
            use_1h_gate=_env_bool("USE_1H_GATE", True),
            use_15m_gate=_env_bool("USE_15M_GATE", True),
            use_vol_gate=_env_bool("USE_VOL_GATE", True),
            max_daily_loss_usd=_env_float("MAX_DAILY_LOSS_USD", 150.0, 0.01),
            max_trades_per_day=_env_int("MAX_TRADES_PER_DAY", 6, 1, 1000),
            preferred_feed=_env_choice("ALPACA_DATA_FEED", "SIP", frozenset({"SIP", "IEX"})),
            auto_entry=_env_bool("AUTO_ENTRY", False),
            paper=_env_bool("ALPACA_PAPER", True),
            http_timeout=_env_float("HTTP_TIMEOUT_SECONDS", 10.0, 1.0, 120.0),
            webhook_secret=_env_str("WEBHOOK_SECRET", ""),
            admin_token=_env_str("ADMIN_TOKEN", ""),
            kill_switch=_env_bool("KILL_SWITCH", False),
            autostart_worker=_env_bool("AUTOSTART_WORKER", True),
            instance_lock_path=_env_str("INSTANCE_LOCK_PATH", "/tmp/spy_brain.lock"),
        )


CFG: Final = Config.from_env()

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

VERSION: Final = "v9.4.0-HARDENED"
NY_TZ: Final = ZoneInfo("America/New_York")

EOD_FLATTEN_MIN: Final = 5
NO_ENTRY_LAST_MIN: Final = 20
ENTRY_START_HHMM: Final = 940
ENTRY_FILL_TIMEOUT: Final = 10.0
WEBHOOK_TTL_SECONDS: Final = 120.0

BAR_MINUTES: Final = 5
BAR_TIMEFRAME: Final = TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute)
LOOKBACK_DAYS: Final = 25
MIN_BARS: Final = 60
RTH_OPEN: Final = dtime(9, 30)
RTH_LAST_BAR: Final = dtime(15, 55)
OHLCV_COLUMNS: Final = ["open", "high", "low", "close", "volume"]
OHLCV_AGG: Final = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

ADX_PERIOD: Final = 14
RSI_PERIOD: Final = 14
EMA_PERIOD: Final = 21
VOL_LOOKBACK: Final = 20
VOL_MIN_RATIO: Final = 0.8
MIN_ATR_PCT: Final = 0.0003
MIN_ATR: Final = 0.05
HTF_EMA_PERIOD: Final = 20
MAX_PRICE_DEVIATION: Final = 0.02
RSI_MIDLINE: Final = 50.0

LIVE_STATUSES: Final = frozenset({"new", "accepted", "pending_new", "partially_filled", "held", "pending_replace"})
TERMINAL_UNFILLED_STATUSES: Final = frozenset({"canceled", "expired", "rejected"})
ENTRY_COID_REGEX: Final = re.compile(r"^spy-(\d{12})-")
WEBHOOK_SIGNALS: Final = {"BUY_LONG": 1, "SELL_SHORT": -1}

NETWORK_ERRORS: Final = (RequestsConnectionError, RequestsTimeout, ProtocolError, ConnectionError, TimeoutError)
BROKER_ERRORS: Final = (*NETWORK_ERRORS, APIError)
RETRYABLE_STATUSES: Final = frozenset({429, 500, 502, 503, 504})
FEED_DENIED_STATUSES: Final = frozenset({403, 422})
API_MAX_ATTEMPTS: Final = 4
API_BACKOFF_START: Final = 0.5
API_BACKOFF_MAX: Final = 10.0

# --------------------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------------------

bot_state: dict[str, Any] = {
    "version": VERSION,
    "last_check": "Not checked yet",
    "last_scan_ms": 0.0,
    "market_open": False,
    "active_window": "Initializing...",
    "entry_gate": "Initializing...",
    "data_feed": CFG.preferred_feed,
    "spy_price": 0.0,
    "last_5m_close": 0.0,
    "session_vwap": 0.0,
    "regime": "Initializing...",
    "signal": "Initializing...",
    "htf_1h_bias": "Initializing...",
    "htf_15m_structure": "Initializing...",
    "vol_gate_ok": False,
    "adx": 0.0,
    "rsi": 0.0,
    "atr": 0.0,
    "risk_mode": CFG.risk_mode,
    "risk_budget_usd": CFG.risk_usd_per_trade,
    "buying_power": 0.0,
    "next_dynamic_qty": CFG.fallback_qty,
    "position": 0,
    "entry_price": 0.0,
    "active_tp": 0.0,
    "active_sl": 0.0,
    "ratchet_locked": False,
    "last_bar_traded": "",
    "cooldown_until": "None",
    "trades_today": 0,
    "day_pnl": 0.0,
    "risk_status": "OK",
    "kill_switch": CFG.kill_switch,
    "last_action": "System booted cleanly",
}
state_lock = threading.Lock()
bot_lock = threading.RLock()
shutdown_event = threading.Event()

_FLAT_FIELDS: Final = {
    "position": 0,
    "entry_price": 0.0,
    "active_tp": 0.0,
    "active_sl": 0.0,
    "ratchet_locked": False,
}


@dataclass(frozen=True)
class PendingSignal:
    signal: str
    received_at: float


@dataclass
class Runtime:
    kill_switch: bool
    last_exit_ts: float = 0.0
    entry_ts: float = 0.0
    trades_day: str = ""
    pending_signal: PendingSignal | None = None
    last_scan_ok: float = field(default_factory=time.time)


runtime = Runtime(kill_switch=CFG.kill_switch)


def update_state(**fields: Any) -> None:
    with state_lock:
        bot_state.update(fields)


def read_state(key: str) -> Any:
    with state_lock:
        return bot_state[key]


def snapshot_state() -> dict[str, Any]:
    with state_lock:
        return dict(bot_state)


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------

T = TypeVar("T")


def enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def status_code_of(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def retry_after_seconds(exc: BaseException) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    return max(0.0, safe_float(headers.get("Retry-After"), 0.0))


def api_call(fn: Callable[..., T], *args: Any, retries: int = API_MAX_ATTEMPTS, **kwargs: Any) -> T:
    """Invoke a broker/data call, retrying transient failures with jittered backoff."""
    delay = API_BACKOFF_START
    name = getattr(fn, "__name__", repr(fn))
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except BROKER_ERRORS as exc:
            transient = isinstance(exc, NETWORK_ERRORS) or status_code_of(exc) in RETRYABLE_STATUSES
            if not transient or attempt >= attempts:
                raise
            wait = min(max(delay, retry_after_seconds(exc)), API_BACKOFF_MAX)
            wait += random.uniform(0.0, wait * 0.25)
            log_event(logging.WARNING, "api_retry", call=name, attempt=attempt, wait_s=round(wait, 2), error=str(exc))
            time.sleep(wait)
            delay *= 2
    raise RuntimeError(f"api_call exhausted retries for {name}")


def secrets_match(supplied: str | None, expected: str) -> bool:
    if not supplied or not expected:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


# --------------------------------------------------------------------------------------
# Broker clients
# --------------------------------------------------------------------------------------

_clients: tuple[TradingClient, StockHistoricalDataClient] | None = None
_clients_lock = threading.Lock()


def get_env_keys() -> tuple[str, str]:
    api_key = (os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_KEY") or "").strip()
    secret_key = (os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET") or "").strip()
    return api_key, secret_key


def _apply_default_timeout(client: object, timeout: float) -> None:
    session = getattr(client, "_session", None)
    original = getattr(session, "request", None)
    if session is None or original is None:
        return

    def request_with_timeout(method: str, url: str, **kwargs: Any) -> Any:
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = timeout
        return original(method, url, **kwargs)

    session.request = request_with_timeout


def get_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    global _clients
    with _clients_lock:
        if _clients is None:
            api_key, secret_key = get_env_keys()
            if not api_key or not secret_key:
                raise RuntimeError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")
            trading = TradingClient(api_key, secret_key, paper=CFG.paper)
            data = StockHistoricalDataClient(api_key, secret_key)
            _apply_default_timeout(trading, CFG.http_timeout)
            _apply_default_timeout(data, CFG.http_timeout)
            _clients = (trading, data)
        return _clients


# --------------------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------------------


class MarketSnapshot(NamedTuple):
    bars: pd.DataFrame
    latest_price: float | None
    feed: str


class MarketData:
    """Incrementally cached 5-minute bars with a one-time SIP to IEX fallback."""

    def __init__(self, preferred_feed: str) -> None:
        self._feed = DataFeed.SIP if preferred_feed == "SIP" else DataFeed.IEX
        self._bars = pd.DataFrame()

    @property
    def feed_name(self) -> str:
        return self._feed.value.upper()

    def _download(self, client: StockHistoricalDataClient, start: datetime) -> pd.DataFrame:
        bars_request = StockBarsRequest(
            symbol_or_symbols=CFG.symbol, timeframe=BAR_TIMEFRAME, start=start, feed=self._feed
        )
        frame = api_call(client.get_stock_bars, bars_request).df
        if frame is None or frame.empty:
            return pd.DataFrame()
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.xs(CFG.symbol, level=0)
        return frame.sort_index()

    def _latest_price(self, client: StockHistoricalDataClient) -> float | None:
        try:
            trades = api_call(
                client.get_stock_latest_trade,
                StockLatestTradeRequest(symbol_or_symbols=CFG.symbol, feed=self._feed),
            )
        except BROKER_ERRORS as exc:
            log_event(logging.DEBUG, "latest_trade_unavailable", error=str(exc))
            return None
        trade = trades.get(CFG.symbol)
        price = safe_float(getattr(trade, "price", None), 0.0)
        if price <= 0:
            return None
        stamp = getattr(trade, "timestamp", None)
        if isinstance(stamp, datetime):
            age = (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds()
            if age > CFG.max_bar_age_seconds:
                return None
        return price

    def refresh(self, client: StockHistoricalDataClient) -> MarketSnapshot:
        full_start = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
        if self._bars.empty:
            start = full_start
        else:
            overlap_start = self._bars.index[-1].to_pydatetime() - timedelta(minutes=2 * BAR_MINUTES)
            start = max(full_start, overlap_start)

        try:
            fresh = self._download(client, start)
        except APIError as exc:
            if self._feed is not DataFeed.SIP or status_code_of(exc) not in FEED_DENIED_STATUSES:
                raise
            log_event(logging.WARNING, "sip_unavailable_falling_back_to_iex", error=str(exc))
            self._feed = DataFeed.IEX
            self._bars = pd.DataFrame()
            fresh = self._download(client, full_start)

        if not fresh.empty:
            merged = fresh if self._bars.empty else pd.concat([self._bars, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            self._bars = merged.loc[merged.index >= pd.Timestamp(full_start)]

        update_state(data_feed=self.feed_name)
        return MarketSnapshot(self._bars, self._latest_price(client), self.feed_name)


market_data = MarketData(CFG.preferred_feed)

# --------------------------------------------------------------------------------------
# Signal engine
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Signals:
    live_price: float
    closed_price: float
    atr: float
    rsi: float
    adx: float
    vwap: float
    ema21: float
    htf_1h_bias: str
    htf_15m_structure: str
    regime: str
    vol_ok: bool
    long_sig: bool
    short_sig: bool
    sig_label: str
    bar_tag: str
    bar_age_seconds: float
    data_ok: bool


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    idx = frame.index
    high, low, close, volume = frame["high"], frame["low"], frame["close"], frame["volume"]
    day = pd.Series(idx.year * 10_000 + idx.month * 100 + idx.day, index=idx)
    session_start = day.ne(day.shift(1))

    prev_close = close.shift(1).where(~session_start, frame["open"])
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    up_move = (high - high.shift(1)).where(~session_start, 0.0)
    down_move = (low.shift(1) - low).where(~session_start, 0.0)
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=idx)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=idx)

    atr = _wilder(true_range, ADX_PERIOD)
    plus_di = 100.0 * _wilder(plus_dm, ADX_PERIOD) / atr
    minus_di = 100.0 * _wilder(minus_dm, ADX_PERIOD) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = _wilder(dx, ADX_PERIOD)

    delta = close.diff().where(~session_start, 0.0)
    avg_gain = _wilder(delta.clip(lower=0.0), RSI_PERIOD)
    avg_loss = _wilder((-delta).clip(lower=0.0), RSI_PERIOD)
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi = (100.0 - 100.0 / (1.0 + rs)).where(avg_loss.isna() | (avg_loss > 0), 100.0)

    typical = (high + low + close) / 3.0
    vwap = (typical * volume).groupby(day).cumsum() / volume.groupby(day).cumsum().replace(0.0, np.nan)

    return frame.assign(
        atr=atr,
        plus_di=plus_di,
        minus_di=minus_di,
        adx=adx,
        rsi=rsi,
        vwap=vwap,
        ema21=close.ewm(span=EMA_PERIOD, adjust=False).mean(),
        vol_avg=volume.rolling(VOL_LOOKBACK).mean(),
    )


def _completed_htf_bars(frame: pd.DataFrame, minutes: int, offset_minutes: int) -> pd.DataFrame:
    """Resample to a higher timeframe and keep only bars fully closed by the last 5m bar."""
    bars = (
        frame[OHLCV_COLUMNS]
        .resample(f"{minutes}min", offset=pd.Timedelta(minutes=offset_minutes))
        .agg(OHLCV_AGG)
        .dropna(subset=["close"])
    )
    cutoff = frame.index[-1] + pd.Timedelta(minutes=BAR_MINUTES)
    return bars.loc[bars.index + pd.Timedelta(minutes=minutes) <= cutoff]


def _hourly_bias(bars: pd.DataFrame) -> str:
    if len(bars) < HTF_EMA_PERIOD + 2:
        return "NEUTRAL"
    ema = bars["close"].ewm(span=HTF_EMA_PERIOD, adjust=False).mean()
    close, ema_now, ema_prev = float(bars["close"].iloc[-1]), float(ema.iloc[-1]), float(ema.iloc[-2])
    if close > ema_now and ema_now >= ema_prev:
        return "BULLISH (Long Only)"
    if close < ema_now and ema_now <= ema_prev:
        return "BEARISH (Short Only)"
    return "NEUTRAL"


def _quarter_hour_structure(bars: pd.DataFrame) -> str:
    if len(bars) < EMA_PERIOD + 1:
        return "MIXED"
    close = bars["close"]
    fast = float(close.ewm(span=8, adjust=False).mean().iloc[-1])
    slow = float(close.ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-1])
    last = float(close.iloc[-1])
    if last > slow and fast > slow:
        return "BULL ALIGNED"
    if last < slow and fast < slow:
        return "BEAR ALIGNED"
    return "MIXED"


def compute_signals(snapshot: MarketSnapshot) -> Signals | None:
    """Compute causal signals from confirmed bars only; returns None on insufficient/invalid data."""
    bars = snapshot.bars
    if bars.empty or not set(OHLCV_COLUMNS).issubset(bars.columns):
        return None

    frame = bars[OHLCV_COLUMNS].tz_convert(NY_TZ).between_time(RTH_OPEN, RTH_LAST_BAR)
    if len(frame) < MIN_BARS:
        return None

    now_ny = pd.Timestamp.now(tz=NY_TZ)
    if frame.index[-1] + pd.Timedelta(minutes=BAR_MINUTES) > now_ny:
        frame = frame.iloc[:-1]
    if len(frame) < MIN_BARS:
        return None

    enriched = _add_indicators(frame)
    last, prev = enriched.iloc[-1], enriched.iloc[-2]
    required = [last[c] for c in ("atr", "adx", "plus_di", "minus_di", "rsi", "vwap", "ema21", "close")]
    if not all(math.isfinite(float(v)) for v in required) or not math.isfinite(float(prev["ema21"])):
        return None

    close, prev_close = float(last["close"]), float(prev["close"])
    atr, adx, rsi = max(float(last["atr"]), MIN_ATR), float(last["adx"]), float(last["rsi"])
    vwap, ema21, prev_ema21 = float(last["vwap"]), float(last["ema21"]), float(prev["ema21"])
    plus_di, minus_di = float(last["plus_di"]), float(last["minus_di"])

    live_price = snapshot.latest_price if snapshot.latest_price else close
    bar_age = (now_ny - (enriched.index[-1] + pd.Timedelta(minutes=BAR_MINUTES))).total_seconds()
    price_sane = close > 0 and abs(live_price / close - 1.0) <= MAX_PRICE_DEVIATION
    data_ok = price_sane and bar_age <= CFG.max_bar_age_seconds

    vol_avg = safe_float(last["vol_avg"], 0.0)
    vol_ok = vol_avg > 0 and float(last["volume"]) >= VOL_MIN_RATIO * vol_avg and atr / close >= MIN_ATR_PCT

    bias = _hourly_bias(_completed_htf_bars(frame, 60, 30))
    structure = _quarter_hour_structure(_completed_htf_bars(frame, 15, 0))

    trending = adx >= CFG.adx_trend_min
    gates_long = (
        (not CFG.use_1h_gate or bias.startswith("BULLISH"))
        and (not CFG.use_15m_gate or structure == "BULL ALIGNED")
        and (not CFG.use_vol_gate or vol_ok)
    )
    gates_short = (
        (not CFG.use_1h_gate or bias.startswith("BEARISH"))
        and (not CFG.use_15m_gate or structure == "BEAR ALIGNED")
        and (not CFG.use_vol_gate or vol_ok)
    )
    long_sig = (
        trending and plus_di > minus_di and close > vwap
        and prev_close <= prev_ema21 and close > ema21
        and rsi > RSI_MIDLINE and gates_long
    )
    short_sig = (
        trending and minus_di > plus_di and close < vwap
        and prev_close >= prev_ema21 and close < ema21
        and rsi < RSI_MIDLINE and gates_short
    )
    label = "LONG SETUP" if long_sig else "SHORT SETUP" if short_sig else "SCANNING"

    return Signals(
        live_price=float(live_price),
        closed_price=close,
        atr=atr,
        rsi=rsi,
        adx=adx,
        vwap=vwap,
        ema21=ema21,
        htf_1h_bias=bias,
        htf_15m_structure=structure,
        regime="TREND" if trending else "CHOP",
        vol_ok=bool(vol_ok),
        long_sig=bool(long_sig),
        short_sig=bool(short_sig),
        sig_label=label,
        bar_tag=enriched.index[-1].strftime("%Y%m%d%H%M"),
        bar_age_seconds=float(bar_age),
        data_ok=bool(data_ok),
    )


def publish_signals(sig: Signals) -> None:
    update_state(
        spy_price=round(sig.live_price, 2),
        last_5m_close=round(sig.closed_price, 2),
        session_vwap=round(sig.vwap, 2),
        regime=sig.regime,
        signal=sig.sig_label,
        htf_1h_bias=sig.htf_1h_bias,
        htf_15m_structure=sig.htf_15m_structure,
        vol_gate_ok=sig.vol_ok,
        adx=round(sig.adx, 2),
        rsi=round(sig.rsi, 2),
        atr=round(sig.atr, 3),
    )


# --------------------------------------------------------------------------------------
# Broker queries
# --------------------------------------------------------------------------------------


def get_position_details(client: TradingClient) -> tuple[int, float]:
    try:
        position = api_call(client.get_open_position, CFG.symbol)
    except APIError as exc:
        if status_code_of(exc) == 404 or "position does not exist" in str(exc).lower():
            return 0, 0.0
        raise
    quantity = int(math.ceil(abs(float(position.qty))))
    signed = -quantity if enum_value(position.side) == "short" else quantity
    return signed, safe_float(position.avg_entry_price, 0.0)


def _flatten_orders(orders: list[Any], live_only: bool = False) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for parent in orders:
        for order in (parent, *(getattr(parent, "legs", None) or ())):
            if order.id in seen:
                continue
            if live_only and enum_value(order.status) not in LIVE_STATUSES:
                continue
            seen.add(order.id)
            result.append(order)
    return result


def get_live_orders(client: TradingClient) -> list[Any]:
    orders_request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[CFG.symbol], nested=True, limit=50)
    return _flatten_orders(api_call(client.get_orders, orders_request), live_only=True)


def fetch_orders_since(client: TradingClient, since: datetime) -> list[Any]:
    orders_request = GetOrdersRequest(
        status=QueryOrderStatus.ALL, symbols=[CFG.symbol], after=since, nested=True, limit=500
    )
    return _flatten_orders(api_call(client.get_orders, orders_request))


def find_exit_legs(client: TradingClient, pos: int) -> tuple[Any | None, Any | None]:
    exit_side = "sell" if pos > 0 else "buy"
    tp_leg: Any | None = None
    sl_leg: Any | None = None
    for order in get_live_orders(client):
        if enum_value(order.side) != exit_side:
            continue
        kind = enum_value(order.order_type)
        if kind == "limit" and tp_leg is None:
            tp_leg = order
        elif kind in ("stop", "stop_limit") and sl_leg is None:
            sl_leg = order
    return tp_leg, sl_leg


def cancel_quietly(client: TradingClient, order_id: Any) -> None:
    try:
        api_call(client.cancel_order_by_id, order_id, retries=1)
    except BROKER_ERRORS as exc:
        log_event(logging.DEBUG, "cancel_skipped", order_id=str(order_id), error=str(exc))


def cancel_symbol_orders(client: TradingClient, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        live = get_live_orders(client)
        if not live:
            return True
        if time.monotonic() >= deadline:
            log_event(logging.WARNING, "cancel_timeout", remaining=len(live))
            return False
        for order in live:
            cancel_quietly(client, order.id)
        time.sleep(0.4)


def _is_entry_fill(order: Any) -> bool:
    coid = getattr(order, "client_order_id", None) or ""
    return bool(ENTRY_COID_REGEX.match(coid)) and safe_float(getattr(order, "filled_qty", None), 0.0) > 0


def format_cooldown(last_exit_ts: float) -> str:
    if last_exit_ts <= 0:
        return "None"
    until = datetime.fromtimestamp(last_exit_ts + CFG.cooldown_minutes * 60, NY_TZ)
    return until.strftime("%I:%M:%S %p %Z")


def cooldown_remaining() -> float:
    return max(0.0, runtime.last_exit_ts + CFG.cooldown_minutes * 60 - time.time())


def sync_daily_counters(client: TradingClient, now_ny: datetime) -> bool:
    """Rebuild trade count and last-exit time from the broker once per trading day."""
    today = now_ny.strftime("%Y-%m-%d")
    if runtime.trades_day == today:
        return True
    midnight = now_ny.replace(hour=0, minute=0, second=0, microsecond=0)
    try:
        orders = fetch_orders_since(client, midnight)
    except BROKER_ERRORS as exc:
        if throttled("daily_sync_failed", 60.0):
            log_event(logging.WARNING, "daily_sync_failed", error=str(exc))
        return False
    fills = [order.filled_at for order in orders if getattr(order, "filled_at", None) is not None]
    if fills:
        runtime.last_exit_ts = max(runtime.last_exit_ts, max(fills).timestamp())
    runtime.trades_day = today
    update_state(
        trades_today=sum(1 for order in orders if _is_entry_fill(order)),
        cooldown_until=format_cooldown(runtime.last_exit_ts),
    )
    return True


def estimate_entry_ts(client: TradingClient) -> float:
    try:
        orders = fetch_orders_since(client, datetime.now(timezone.utc) - timedelta(days=5))
    except BROKER_ERRORS as exc:
        log_event(logging.WARNING, "entry_ts_estimate_failed", error=str(exc))
        return time.time()
    stamps = [o.filled_at.timestamp() for o in orders if getattr(o, "filled_at", None) and _is_entry_fill(o)]
    return max(stamps, default=time.time())


# --------------------------------------------------------------------------------------
# Position state
# --------------------------------------------------------------------------------------


def reset_position_state(**extra: Any) -> None:
    update_state(**_FLAT_FIELDS, **extra)


def record_exit(reason: str) -> None:
    runtime.last_exit_ts = time.time()
    runtime.entry_ts = 0.0
    reset_position_state(
        cooldown_until=format_cooldown(runtime.last_exit_ts),
        last_action=f"Exited ({reason})",
    )
    log_event(logging.INFO, "position_exit", reason=reason)


def _leg_price(order: Any | None, attribute: str) -> float:
    if order is None:
        return 0.0
    return round(safe_float(getattr(order, attribute, None), 0.0), 2)


def sync_open_position(
    client: TradingClient, pos: int, avg_entry: float, tp_leg: Any | None, sl_leg: Any | None
) -> None:
    """Mirror broker truth into local state (also rehydrates after a restart)."""
    direction = 1 if pos > 0 else -1
    entry_px = round(avg_entry, 2)
    active_sl = _leg_price(sl_leg, "stop_price")
    locked = bool(read_state("ratchet_locked")) or (active_sl > 0 and (active_sl - entry_px) * direction > 0)
    update_state(
        position=pos,
        entry_price=entry_px,
        active_tp=_leg_price(tp_leg, "limit_price"),
        active_sl=active_sl,
        ratchet_locked=locked,
    )
    if runtime.entry_ts <= 0:
        runtime.entry_ts = estimate_entry_ts(client)


def rearm_protective_stop(client: TradingClient, pos: int) -> None:
    stop_price = float(read_state("active_sl"))
    if stop_price <= 0:
        return
    try:
        api_call(
            client.submit_order,
            StopOrderRequest(
                symbol=CFG.symbol,
                qty=abs(pos),
                side=OrderSide.SELL if pos > 0 else OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
                stop_price=stop_price,
                client_order_id=f"spy-rearm-{int(time.time())}",
            ),
            retries=1,
        )
        log_event(logging.WARNING, "protective_stop_rearmed", position=pos, stop=stop_price)
    except BROKER_ERRORS as exc:
        log_event(logging.CRITICAL, "protective_stop_rearm_failed", position=pos, error=str(exc))


def flatten_position(client: TradingClient, reason: str) -> str:
    with bot_lock:
        if not cancel_symbol_orders(client):
            log_event(logging.ERROR, "flatten_cancel_incomplete", reason=reason)
        pos, _ = get_position_details(client)
        if pos == 0:
            runtime.entry_ts = 0.0
            reset_position_state()
            return "Already flat"

        log_event(logging.INFO, "flatten", position=pos, reason=reason)
        try:
            api_call(client.close_position, CFG.symbol)
        except BROKER_ERRORS as exc:
            log_event(logging.CRITICAL, "flatten_failed", reason=reason, error=str(exc))
            rearm_protective_stop(client, pos)
            raise
        record_exit(reason)
        return "Flattened"


# --------------------------------------------------------------------------------------
# Sizing and entries
# --------------------------------------------------------------------------------------


def compute_brackets(entry_px: float, direction: int, atr: float) -> tuple[float, float]:
    take_profit = round(entry_px + direction * atr * CFG.tp_mult, 2)
    stop_loss = round(entry_px - direction * atr * CFG.sl_mult, 2)
    return take_profit, stop_loss


def calculate_dynamic_qty(
    entry_px: float, sl_px: float, equity: float, buying_power: float = 0.0
) -> tuple[int, float]:
    stop_distance = max(abs(entry_px - sl_px), 0.01)
    if CFG.risk_mode == "PCT_EQUITY" and equity > 0:
        risk_budget = equity * (CFG.risk_pct_per_trade / 100.0)
    else:
        risk_budget = CFG.risk_usd_per_trade
    shares = risk_budget / stop_distance
    if equity > 0 and entry_px > 0:
        max_notional = equity * (CFG.max_position_pct_equity / 100.0)
        if buying_power > 0:
            max_notional = min(max_notional, buying_power * 0.95)
        shares = min(shares, max_notional / entry_px)
    return max(0, int(math.floor(shares))), round(risk_budget, 2)


def build_entry_coid(bar_tag: str, direction: int, manual: bool) -> str:
    return f"spy-{bar_tag}-{'L' if direction > 0 else 'S'}{'M' if manual else 'A'}"


def submit_idempotent(client: TradingClient, order_request: MarketOrderRequest) -> Any:
    """Submit once; on an ambiguous failure, resolve by client order id before giving up."""
    try:
        return api_call(client.submit_order, order_request, retries=1)
    except BROKER_ERRORS as exc:
        log_event(logging.WARNING, "submit_ambiguous", coid=order_request.client_order_id, error=str(exc))
        try:
            return api_call(client.get_order_by_client_id, order_request.client_order_id)
        except BROKER_ERRORS:
            raise exc


def wait_for_fill(client: TradingClient, order_id: Any, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status = enum_value(api_call(client.get_order_by_id, order_id, retries=2).status)
        except BROKER_ERRORS as exc:
            log_event(logging.DEBUG, "fill_poll_error", error=str(exc))
        else:
            if status == "filled":
                return True
            if status in TERMINAL_UNFILLED_STATUSES:
                return False
        time.sleep(0.25)
    return False


def await_exit_legs(client: TradingClient, pos: int, attempts: int = 4) -> tuple[Any | None, Any | None]:
    tp_leg: Any | None = None
    sl_leg: Any | None = None
    for attempt in range(attempts):
        tp_leg, sl_leg = find_exit_legs(client, pos)
        if tp_leg is not None and sl_leg is not None:
            break
        if attempt < attempts - 1:
            time.sleep(0.4)
    return tp_leg, sl_leg


def enter_bracket_trade(
    client: TradingClient,
    sig: Signals,
    direction: int,
    label: str,
    equity: float,
    buying_power: float,
    manual: bool = False,
) -> str:
    with bot_lock:
        current_pos, _ = get_position_details(client)
        if current_pos != 0:
            return "Already in position"
        if not cancel_symbol_orders(client):
            return "Stale orders not cleared"

        price = sig.live_price
        take_profit, stop_loss = compute_brackets(price, direction, sig.atr)
        qty, risk_budget = calculate_dynamic_qty(price, stop_loss, equity, buying_power)
        update_state(next_dynamic_qty=qty, risk_budget_usd=risk_budget)
        if qty < 1 or stop_loss <= 0 or take_profit <= 0:
            log_event(logging.WARNING, "entry_skipped_invalid_size", qty=qty, tp=take_profit, sl=stop_loss)
            return "Size below minimum"

        coid = build_entry_coid(sig.bar_tag, direction, manual)
        update_state(last_bar_traded=sig.bar_tag)
        order_request = MarketOrderRequest(
            symbol=CFG.symbol,
            qty=qty,
            side=OrderSide.BUY if direction > 0 else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit),
            stop_loss=StopLossRequest(stop_price=stop_loss),
            client_order_id=coid,
        )
        try:
            parent = submit_idempotent(client, order_request)
        except BROKER_ERRORS as exc:
            log_event(logging.ERROR, "entry_rejected", label=label, coid=coid, error=str(exc))
            return "Entry rejected"

        if not wait_for_fill(client, parent.id, ENTRY_FILL_TIMEOUT):
            cancel_quietly(client, parent.id)

        pos, avg_entry = get_position_details(client)
        if pos == 0:
            cancel_symbol_orders(client)
            log_event(logging.INFO, "entry_no_fill", label=label, coid=coid)
            return "No fill"

        runtime.entry_ts = time.time()
        update_state(trades_today=int(read_state("trades_today")) + 1)

        tp_leg, sl_leg = await_exit_legs(client, pos)
        if tp_leg is None or sl_leg is None:
            log_event(logging.CRITICAL, "entry_missing_bracket_legs", position=pos, coid=coid)
            flatten_position(client, "Missing Bracket Legs on Entry")
            return "Emergency flattened"

        active_tp, active_sl = _leg_price(tp_leg, "limit_price"), _leg_price(sl_leg, "stop_price")
        leg_qty = safe_float(getattr(sl_leg, "qty", None), float(abs(pos)))
        if int(leg_qty) != abs(pos):
            log_event(logging.WARNING, "leg_quantity_mismatch", position=pos, leg_qty=leg_qty)

        update_state(
            position=pos,
            entry_price=round(avg_entry, 2),
            active_tp=active_tp,
            active_sl=active_sl,
            ratchet_locked=False,
            last_action=f"{label} {pos:+d} sh @ {avg_entry:.2f} (Risk ${risk_budget:.0f})",
        )
        log_event(
            logging.INFO, "entered", label=label, symbol=CFG.symbol, position=pos,
            fill=round(avg_entry, 2), tp=active_tp, sl=active_sl, coid=coid,
            slippage=round(abs(avg_entry - price), 4),
        )
        return "Order sent"


def manage_open_position(
    client: TradingClient, pos: int, tp_leg: Any | None, sl_leg: Any | None, live_price: float, atr: float
) -> None:
    if read_state("ratchet_locked") or tp_leg is None or sl_leg is None:
        return

    direction = 1 if pos > 0 else -1
    entry_px = float(read_state("entry_price"))
    tp_price, sl_price = safe_float(tp_leg.limit_price, 0.0), safe_float(sl_leg.stop_price, 0.0)
    if entry_px <= 0 or tp_price <= 0 or sl_price <= 0:
        return

    target_distance = abs(tp_price - entry_px)
    if (live_price - entry_px) * direction < target_distance * CFG.ratchet_trig:
        return

    new_sl = round(entry_px + direction * atr * CFG.ratchet_lock, 2)
    if (new_sl - sl_price) * direction <= 0:
        update_state(ratchet_locked=True)
        return
    if (live_price - new_sl) * direction < 0.05:
        return

    try:
        api_call(client.replace_order_by_id, sl_leg.id, ReplaceOrderRequest(stop_price=new_sl))
    except BROKER_ERRORS as exc:
        log_event(logging.WARNING, "ratchet_replace_failed", error=str(exc))
        return
    update_state(active_sl=new_sl, ratchet_locked=True, last_action=f"Ratchet Locked SL @ {new_sl:.2f}")
    log_event(logging.INFO, "ratchet_triggered", new_sl=new_sl)


# --------------------------------------------------------------------------------------
# Risk gate and entry gating
# --------------------------------------------------------------------------------------


class RiskCheck(NamedTuple):
    ok: bool
    message: str
    equity: float
    buying_power: float


class SessionWindow(NamedTuple):
    label: str
    entry_open: bool
    eod_flatten: bool


def check_risk_gate(client: TradingClient) -> RiskCheck:
    account = api_call(client.get_account)
    equity = safe_float(account.equity)
    buying_power = safe_float(getattr(account, "buying_power", None), equity) or equity
    day_pnl = equity - safe_float(account.last_equity, equity)
    update_state(day_pnl=round(day_pnl, 2), buying_power=round(buying_power, 2))

    if runtime.kill_switch:
        message = "Kill switch active"
    elif equity <= 0:
        message = "Invalid account equity"
    elif getattr(account, "trading_blocked", False) or getattr(account, "account_blocked", False):
        message = "Account trading blocked"
    elif day_pnl <= -CFG.max_daily_loss_usd:
        message = f"Daily loss cap hit (${day_pnl:.2f})"
    else:
        message = "OK"
    update_state(risk_status=message)
    return RiskCheck(message == "OK", message, equity, buying_power)


def classify_window(now_ny: datetime, mins_to_close: float) -> SessionWindow:
    hhmm = now_ny.hour * 100 + now_ny.minute
    if mins_to_close <= EOD_FLATTEN_MIN:
        return SessionWindow("EOD flatten", False, True)
    if mins_to_close < NO_ENTRY_LAST_MIN:
        return SessionWindow("No new entries", False, False)
    if hhmm < ENTRY_START_HHMM:
        return SessionWindow("Pre-entry warmup", False, False)
    return SessionWindow("Entry window", True, False)


def entry_block_reason(window: SessionWindow, risk: RiskCheck, daily_synced: bool, sig: Signals) -> str | None:
    if not window.entry_open:
        return window.label
    if not risk.ok:
        return risk.message
    if not daily_synced:
        return "Daily counters unsynced"
    if not sig.data_ok:
        return "Stale or invalid market data"
    if int(read_state("trades_today")) >= CFG.max_trades_per_day:
        return "Max trades per day reached"
    remaining = cooldown_remaining()
    if remaining > 0:
        return f"Cooldown ({remaining:.0f}s left)"
    return None


def peek_pending_signal() -> PendingSignal | None:
    with state_lock:
        pending = runtime.pending_signal
        if pending is not None and time.time() - pending.received_at > WEBHOOK_TTL_SECONDS:
            runtime.pending_signal = None
            return None
        return pending


def clear_pending_signal() -> None:
    with state_lock:
        runtime.pending_signal = None


def select_entry(sig: Signals) -> tuple[int, str, bool] | None:
    pending = peek_pending_signal()
    if pending is not None:
        clear_pending_signal()
        return WEBHOOK_SIGNALS[pending.signal], f"Webhook {pending.signal}", True
    if CFG.auto_entry and sig.bar_tag != read_state("last_bar_traded"):
        if sig.long_sig:
            return 1, "Auto LONG", False
        if sig.short_sig:
            return -1, "Auto SHORT", False
    return None


# --------------------------------------------------------------------------------------
# Scan loop
# --------------------------------------------------------------------------------------


def _manage_open_cycle(
    trade_client: TradingClient,
    data_client: StockHistoricalDataClient,
    pos: int,
    avg_entry: float,
    risk: RiskCheck,
    window: SessionWindow,
) -> None:
    """Safety exits run before (and independent of) any market-data access."""
    tp_leg, sl_leg = find_exit_legs(trade_client, pos)
    sync_open_position(trade_client, pos, avg_entry, tp_leg, sl_leg)

    if window.eod_flatten:
        flatten_position(trade_client, "EOD Flatten")
        return
    if not risk.ok:
        flatten_position(trade_client, f"Risk Halt ({risk.message})")
        return
    held_minutes = (time.time() - runtime.entry_ts) / 60.0
    if held_minutes >= CFG.max_hold_minutes:
        flatten_position(trade_client, f"Max Hold ({held_minutes:.0f}m)")
        return
    if sl_leg is None:
        log_event(logging.CRITICAL, "position_without_stop", position=pos)
        flatten_position(trade_client, "Missing Stop Leg")
        return
    if read_state("ratchet_locked") or tp_leg is None:
        return

    sig = compute_signals(market_data.refresh(data_client))
    if sig is None or not sig.data_ok:
        return
    publish_signals(sig)
    manage_open_position(trade_client, pos, tp_leg, sl_leg, sig.live_price, sig.atr)


def _scan_for_entry(
    trade_client: TradingClient,
    data_client: StockHistoricalDataClient,
    window: SessionWindow,
    risk: RiskCheck,
    daily_synced: bool,
) -> None:
    sig = compute_signals(market_data.refresh(data_client))
    if sig is None:
        update_state(entry_gate="Insufficient market data")
        return
    publish_signals(sig)

    reason = entry_block_reason(window, risk, daily_synced, sig)
    update_state(entry_gate=reason or "OPEN")
    if reason is not None:
        return

    selection = select_entry(sig)
    if selection is None:
        return
    direction, label, manual = selection
    result = enter_bracket_trade(
        trade_client, sig, direction, label, risk.equity, risk.buying_power, manual=manual
    )
    log_event(logging.INFO, "entry_attempt", label=label, result=result, bar_tag=sig.bar_tag)


def _scan_once() -> None:
    now_ny = datetime.now(NY_TZ)
    update_state(last_check=now_ny.strftime("%Y-%m-%d %I:%M:%S %p %Z"))
    trade_client, data_client = get_clients()

    clock = api_call(trade_client.get_clock)
    is_open = bool(clock.is_open)
    update_state(market_open=is_open)
    if not is_open:
        update_state(active_window="Market closed")
        return

    mins_to_close = (clock.next_close - clock.timestamp).total_seconds() / 60.0
    window = classify_window(now_ny, mins_to_close)
    update_state(active_window=window.label)

    daily_synced = sync_daily_counters(trade_client, now_ny)
    pos, avg_entry = get_position_details(trade_client)
    risk = check_risk_gate(trade_client)

    if pos != 0:
        _manage_open_cycle(trade_client, data_client, pos, avg_entry, risk, window)
        return
    if read_state("position") != 0:
        record_exit("Broker-side exit")
    _scan_for_entry(trade_client, data_client, window, risk, daily_synced)


def run_single_brain_scan() -> None:
    if not bot_lock.acquire(blocking=False):
        return
    started = time.monotonic()
    try:
        _scan_once()
        runtime.last_scan_ok = time.time()
    except Exception as exc:  # the scan loop must survive any single-cycle failure
        signature = f"{type(exc).__name__}:{exc}"
        if throttled(f"scan_error:{signature}", 300.0):
            log.error("scan_error", exc_info=True, extra={"fields": {"error": str(exc)}})
        update_state(last_action=f"Error: {str(exc)[:120]}")
    finally:
        update_state(last_scan_ms=round((time.monotonic() - started) * 1000.0, 1))
        bot_lock.release()


def background_loop() -> None:
    while not shutdown_event.is_set():
        try:
            run_single_brain_scan()
        except Exception:
            log.exception("scan_loop_error")
        shutdown_event.wait(CFG.check_seconds)


# --------------------------------------------------------------------------------------
# Worker lifecycle
# --------------------------------------------------------------------------------------

_worker_thread: threading.Thread | None = None
_worker_start_lock = threading.Lock()
_instance_lock_handle: Any = None


def acquire_instance_lock() -> bool:
    """Prevent two bot loops on one host (e.g. multiple Gunicorn workers)."""
    global _instance_lock_handle
    if _instance_lock_handle is not None or fcntl is None:
        return True
    try:
        handle = open(CFG.instance_lock_path, "a+")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        log_event(logging.ERROR, "instance_lock_unavailable", path=CFG.instance_lock_path, error=str(exc))
        return False
    _instance_lock_handle = handle
    return True


def worker_alive() -> bool:
    return _worker_thread is not None and _worker_thread.is_alive()


def start_worker() -> bool:
    global _worker_thread
    with _worker_start_lock:
        if worker_alive():
            return True
        if not acquire_instance_lock():
            return False
        _worker_thread = threading.Thread(target=background_loop, daemon=True, name="spy-brain-worker")
        _worker_thread.start()
        log_event(logging.INFO, "worker_started", version=VERSION, symbol=CFG.symbol, paper=CFG.paper)
        return True


def is_healthy() -> bool:
    return worker_alive() and time.time() - runtime.last_scan_ok <= max(3 * CFG.check_seconds, 90)


atexit.register(shutdown_event.set)

# --------------------------------------------------------------------------------------
# HTTP surface
# --------------------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024


def _header_or_bearer(header_name: str) -> str | None:
    supplied = request.headers.get(header_name)
    if supplied:
        return supplied.strip()
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def is_admin_request() -> bool:
    return secrets_match(_header_or_bearer("X-Admin-Token"), CFG.admin_token)


@app.get("/")
def home() -> tuple[Any, int]:
    body: dict[str, Any] = {
        "status": f"SPY Autonomous Brain {VERSION} is Alive!",
        "healthy": is_healthy(),
    }
    if is_admin_request():
        body["live_dashboard"] = snapshot_state()
    return jsonify(body), 200


@app.get("/healthz")
def healthz() -> tuple[Any, int]:
    healthy = is_healthy()
    return jsonify(
        healthy=healthy,
        worker_alive=worker_alive(),
        seconds_since_last_ok_scan=round(time.time() - runtime.last_scan_ok, 1),
    ), (200 if healthy else 503)


@app.post("/webhook")
def webhook() -> tuple[Any, int]:
    if not CFG.webhook_secret:
        return jsonify(error="webhook disabled"), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="invalid json body"), 400
    supplied = request.headers.get("X-Webhook-Secret") or payload.get("secret")
    if not isinstance(supplied, str) or not secrets_match(supplied.strip(), CFG.webhook_secret):
        return jsonify(error="unauthorized"), 401
    signal = str(payload.get("signal", "")).strip().upper()
    if signal not in WEBHOOK_SIGNALS:
        return jsonify(error="unsupported signal", allowed=sorted(WEBHOOK_SIGNALS)), 400
    if not worker_alive():
        return jsonify(error="trading worker not running in this process"), 503
    with state_lock:
        runtime.pending_signal = PendingSignal(signal=signal, received_at=time.time())
    log_event(logging.INFO, "webhook_queued", signal=signal)
    return jsonify(status="queued", signal=signal, ttl_seconds=WEBHOOK_TTL_SECONDS), 202


@app.post("/kill")
def kill_switch_route() -> tuple[Any, int]:
    if not CFG.admin_token:
        return jsonify(error="admin disabled"), 503
    if not is_admin_request():
        return jsonify(error="unauthorized"), 401
    payload = request.get_json(silent=True)
    active = True
    if isinstance(payload, dict) and "active" in payload:
        if not isinstance(payload["active"], bool):
            return jsonify(error="'active' must be a boolean"), 400
        active = payload["active"]
    runtime.kill_switch = active
    update_state(kill_switch=active)
    log_event(logging.WARNING, "kill_switch_set", active=active)
    return jsonify(kill_switch=active), 200


if CFG.autostart_worker:
    start_worker()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
