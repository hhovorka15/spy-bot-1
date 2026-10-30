import hmac
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from flask import Flask, jsonify, request
from alpaca.common.enums import Sort
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass,
    OrderSide,
    PositionSide,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    ReplaceOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spy-brain-v9.1")

# =============================================================================
# 1. MASTER CONFIGURATION
# =============================================================================
SYMBOL = os.environ.get("TRADE_SYMBOL", "SPY").strip().upper()
QTY = int(os.environ.get("TRADE_QTY", "10"))
CHECK_SECONDS = int(os.environ.get("CHECK_SECONDS", "30"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "15"))
MAX_HOLD_MINUTES = int(os.environ.get("MAX_HOLD_MINUTES", "90"))
MAX_BAR_AGE_SECONDS = int(os.environ.get("MAX_BAR_AGE_SECONDS", "360"))

ADX_RANGE_MAX = 20.0
ADX_TREND_MIN = 25.0
MR_BAND_MULT = 1.60
MR_MIN_TP_MULT = 1.20
MR_SL_MULT = 1.50
TB_TP_MULT = 2.25
TB_SL_MULT = 1.25
RATCHET_TRIG = 0.75
RATCHET_LOCK = 0.35

DB_PATH = os.environ.get("STATE_DB_PATH", "/tmp/brain_v9_1.db")
MAX_DAILY_LOSS_USD = float(os.environ.get("MAX_DAILY_LOSS_USD", "150.0"))
MAX_TRADES_PER_DAY = int(os.environ.get("MAX_TRADES_PER_DAY", "6"))
RTH_ONLY = os.environ.get("RTH_ONLY", "true").lower() == "true"
PREFERRED_FEED = os.environ.get("ALPACA_DATA_FEED", "SIP").strip().upper()

LEASE_TTL = 90
ENTRY_FILL_TIMEOUT = 10.0
EOD_FLATTEN_MIN = 5
NO_ENTRY_LAST_MIN = 20
PENDING_TTL = 120
LIVE_STATUSES = {
    "new",
    "accepted",
    "pending_new",
    "partially_filled",
    "held",
    "pending_replace",
    "accepted_for_bidding",
    "calculated",
}
INSTANCE_ID = f"{os.getpid()}-{uuid.uuid4().hex[:6]}"

app = Flask(__name__)
order_lock = threading.RLock()
scan_lock = threading.Lock()
ny_tz = ZoneInfo("America/New_York")

bot_state = {
    "version": "v9.1-STATELESS-PROD",
    "instance_id": INSTANCE_ID,
    "last_check": "Not checked yet",
    "scan_step": "Idle",
    "market_open": False,
    "active_window": "Checking...",
    "data_feed": PREFERRED_FEED,
    "spy_price": 0.0,
    "last_5m_close": 0.0,
    "closed_bar_age_s": 0.0,
    "session_vwap": 0.0,
    "regime": "Initializing...",
    "adx": 0.0,
    "rsi": 0.0,
    "atr": 0.0,
    "position": 0,
    "entry_price": 0.0,
    "active_tp": 0.0,
    "active_sl": 0.0,
    "ratchet_locked": False,
    "cooldown_until": "None",
    "trades_today": 0,
    "last_action": "System booted cleanly",
    "risk_status": "OK",
    "day_pnl": 0.0,
    "leader": False,
}
worker_started = False
scan_errors = 0

# =============================================================================
# 2. LOCAL CACHE & JOURNAL STORE (BACKED BY BROKER RECONCILIATION)
# =============================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lease (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    owner TEXT NOT NULL,
    expires REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    side INTEGER NOT NULL,
    qty INTEGER NOT NULL,
    regime TEXT,
    adx REAL,
    rsi REAL,
    atr REAL,
    vwap REAL,
    entry_px REAL,
    tp_px REAL,
    sl_px REAL,
    ratcheted INTEGER DEFAULT 0,
    exit_px REAL,
    exit_reason TEXT,
    pnl REAL,
    bar_id TEXT
);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        with self._db() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    def get(self, k: str, default=None):
        with self._db() as c:
            r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else default

    def set(self, k: str, v):
        with self._db() as c:
            c.execute(
                "INSERT INTO kv(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (k, json.dumps(v)),
            )

    def trade_open(self, ctx: dict, sig: dict | None) -> int:
        with self._db() as c:
            cur = c.execute(
                "INSERT INTO trades(day, opened_at, side, qty, regime, adx, rsi, atr, vwap, entry_px, tp_px, sl_px, bar_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ctx["day"],
                    ctx["opened_at"],
                    ctx["side"],
                    ctx["qty"],
                    ctx["regime"],
                    sig["adx"] if sig else None,
                    sig["rsi"] if sig else None,
                    ctx["atr"],
                    sig["vwap"] if sig else None,
                    ctx["entry_px"],
                    ctx["tp"],
                    ctx["sl"],
                    ctx.get("bar_id"),
                ),
            )
            return int(cur.lastrowid)

    def trade_close(
        self,
        trade_id: int | None,
        exit_px: float | None,
        reason: str,
        pnl: float | None,
        ratcheted: bool,
    ):
        if not trade_id:
            return
        with self._db() as c:
            c.execute(
                "UPDATE trades SET closed_at=?, exit_px=?, exit_reason=?, pnl=?, ratcheted=? WHERE id=?",
                (
                    datetime.now(timezone.utc).isoformat(),
                    exit_px,
                    reason,
                    pnl,
                    int(bool(ratcheted)),
                    trade_id,
                ),
            )

    def trades_today(self, day: str) -> int:
        with self._db() as c:
            return int(
                c.execute("SELECT COUNT(*) AS n FROM trades WHERE day=?", (day,)).fetchone()["n"]
            )

    def recent_trades(self, n: int = 200) -> list[dict]:
        with self._db() as c:
            return [
                dict(r)
                for r in c.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (n,))
            ]

    def acquire_lease(self, owner: str, ttl: int) -> bool:
        now = time.time()
        c = sqlite3.connect(self.path, timeout=10)
        try:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT owner, expires FROM lease WHERE id=1").fetchone()
            if row is None or row[0] == owner or row[1] < now:
                c.execute(
                    "INSERT INTO lease(id, owner, expires) VALUES (1,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET owner=excluded.owner, expires=excluded.expires",
                    (owner, now + ttl),
                )
                c.commit()
                return True
            c.rollback()
            return False
        except Exception:
            c.rollback()
            return False
        finally:
            c.close()


store = Store(DB_PATH)

# =============================================================================
# 3. API WRAPPERS, BACKOFF & NOTIFICATION HELPERS
# =============================================================================
def _v(x) -> str:
    return str(getattr(x, "value", x)).lower()


def notify(msg: str):
    log.warning("ALERT: %s", msg)
    tok = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not (tok and chat):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            json={"chat_id": chat, "text": f"[SPY Brain v9.1] {msg}"},
            timeout=5,
        )
    except Exception as e:
        log.error("Telegram notification failed: %s", e)


def get_env_keys() -> tuple[str, str, bool]:
    api_key = (
        os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_KEY") or ""
    ).strip()
    secret_key = (
        os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET") or ""
    ).strip()
    paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
    return api_key, secret_key, paper


_clients = None


def get_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    global _clients
    if _clients is None:
        api_key, secret_key, paper = get_env_keys()
        if not api_key or not secret_key:
            raise RuntimeError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")
        _clients = (
            TradingClient(api_key, secret_key, paper=paper),
            StockHistoricalDataClient(api_key, secret_key),
        )
    return _clients


def api_call(fn, *args, retries: int = 3, **kwargs):
    delay = 0.5
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except APIError as e:
            code = getattr(e, "status_code", None)
            if code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                log.warning("Transient APIError (%s) in %s; retrying in %.1fs", code, fn.__name__, delay)
                time.sleep(delay)
                delay *= 2
                continue
            raise


def get_position_details(client: TradingClient) -> tuple[int, float]:
    try:
        pos = api_call(client.get_open_position, SYMBOL)
        qty = abs(int(float(pos.qty)))
        signed_qty = -qty if pos.side == PositionSide.SHORT else qty
        return signed_qty, float(pos.avg_entry_price)
    except APIError as e:
        if getattr(e, "status_code", None) == 404 or "position does not exist" in str(e).lower():
            return 0, 0.0
        log.error("APIError in get_position_details: %s", e)
        raise


def fetch_bars_with_fallback(data_client: StockHistoricalDataClient) -> pd.DataFrame:
    start_time = datetime.now(timezone.utc) - timedelta(days=7)
    feed_enum = DataFeed.SIP if PREFERRED_FEED == "SIP" else DataFeed.IEX
    try:
        req = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_time,
            feed=feed_enum,
        )
        bars = api_call(data_client.get_stock_bars, req).df
        bot_state["data_feed"] = feed_enum.value.upper()
    except APIError as e:
        if feed_enum == DataFeed.SIP:
            log.warning("SIP feed unavailable (%s); falling back to IEX", e)
            req = StockBarsRequest(
                symbol_or_symbols=SYMBOL,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start_time,
                feed=DataFeed.IEX,
            )
            bars = api_call(data_client.get_stock_bars, req).df
            bot_state["data_feed"] = "IEX (Fallback)"
        else:
            raise

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(SYMBOL, level=0)
    bars = bars.sort_index()

    try:
        t_req = StockLatestTradeRequest(
            symbol_or_symbols=SYMBOL,
            feed=DataFeed.IEX if "IEX" in bot_state["data_feed"] else DataFeed.SIP,
        )
        latest = api_call(data_client.get_stock_latest_trade, t_req)
        if SYMBOL in latest and latest[SYMBOL].price:
            bars.attrs["latest_trade_price"] = float(latest[SYMBOL].price)
    except APIError as e:
        log.debug("Could not fetch latest trade price: %s", e)

    return bars


# =============================================================================
# 4. STATELESS BROKER RECONCILIATION & OCO LAYER
# =============================================================================
def recent_orders(client: TradingClient, after: datetime | None = None):
    after = after or (datetime.now(timezone.utc) - timedelta(days=5))
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        symbols=[SYMBOL],
        after=after,
        nested=True,
        limit=100,
        direction=Sort.DESC,
    )
    return api_call(client.get_orders, req)


def broker_session_metrics(client: TradingClient, now_ny: datetime) -> tuple[int, float]:
    """Reconstructs today's entry count and latest exit timestamp directly from Alpaca orders."""
    session_start = datetime.combine(now_ny.date(), dtime(0, 0), tzinfo=ny_tz).astimezone(timezone.utc)
    orders = recent_orders(client, after=session_start)
    entries = 0
    latest_exit_ts = float(store.get("last_exit_ts", 0.0))

    for o in orders:
        coid = str(getattr(o, "client_order_id", "") or "")
        if coid.startswith("spy9-") and o.filled_at:
            entries += 1
        for leg in getattr(o, "legs", None) or []:
            if leg.filled_at and float(leg.filled_qty or 0) > 0:
                latest_exit_ts = max(latest_exit_ts, leg.filled_at.timestamp())

    if latest_exit_ts > float(store.get("last_exit_ts", 0.0)):
        store.set("last_exit_ts", latest_exit_ts)

    total_today = max(entries, store.trades_today(now_ny.strftime("%Y-%m-%d")))
    bot_state["trades_today"] = total_today
    return total_today, latest_exit_ts


def live_orders(client: TradingClient) -> list:
    out = []
    for o in recent_orders(client):
        for x in [o, *(getattr(o, "legs", None) or [])]:
            if _v(x.status) in LIVE_STATUSES:
                out.append(x)
    return out


def find_exit_legs(client: TradingClient) -> tuple:
    tp = sl = None
    for o in live_orders(client):
        t = _v(o.order_type)
        if t == "limit" and tp is None:
            tp = o
        elif t in ("stop", "stop_limit") and sl is None:
            sl = o
    return tp, sl


def cancel_symbol_orders(client: TradingClient, timeout: float = 6.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        live = live_orders(client)
        if not live:
            return True
        for o in live:
            try:
                api_call(client.cancel_order_by_id, o.id, retries=1)
            except APIError:
                pass
        time.sleep(0.4)
    return not live_orders(client)


def wait_fill(client: TradingClient, order_id, timeout: float = 10.0):
    end = time.time() + timeout
    o = None
    while time.time() < end:
        try:
            o = api_call(client.get_order_by_id, order_id)
            if _v(o.status) in ("filled", "canceled", "expired", "rejected"):
                return o
        except APIError as e:
            log.warning("APIError polling order %s: %s", order_id, e)
        time.sleep(0.25)
    return o


def place_oco(client: TradingClient, signed_qty: int, tp: float, sl: float):
    tp, sl = round(tp, 2), round(sl, 2)
    req = LimitOrderRequest(
        symbol=SYMBOL,
        qty=abs(signed_qty),
        side=OrderSide.SELL if signed_qty > 0 else OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
        limit_price=tp,
        order_class=OrderClass.OCO,
        take_profit=TakeProfitRequest(limit_price=tp),
        stop_loss=StopLossRequest(stop_price=sl),
    )
    return api_call(client.submit_order, req)


def finalize_trade(ctx: dict, exit_px: float | None, reason: str):
    pnl = (exit_px - ctx["entry_px"]) * ctx["side"] * ctx["qty"] if exit_px else None
    store.trade_close(ctx.get("trade_id"), exit_px, reason, pnl, bool(ctx.get("ratcheted")))
    store.set("trade_ctx", None)
    store.set("last_exit_ts", time.time())
    bot_state["last_action"] = f"Exited ({reason})"
    notify(f"EXIT {reason}: px={exit_px} | PnL=${None if pnl is None else round(pnl, 2)}")


def settle_closed_trade(client: TradingClient, ctx: dict, default_reason: str = "Broker Leg Fill"):
    exit_px, reason = None, default_reason
    try:
        opened = datetime.fromisoformat(ctx["opened_at"]) - timedelta(seconds=5)
        exit_side = "sell" if ctx["side"] > 0 else "buy"
        fills = []
        for o in recent_orders(client, after=opened):
            for x in [o, *(getattr(o, "legs", None) or [])]:
                if _v(x.side) == exit_side and x.filled_at and float(x.filled_qty or 0) > 0:
                    fills.append(x)
        if fills:
            x = max(fills, key=lambda f: f.filled_at)
            exit_px = float(x.filled_avg_price)
            t = _v(x.order_type)
            if t == "limit":
                reason = "Take-Profit Hit"
            elif t.startswith("stop"):
                reason = "Ratchet Stop Hit" if ctx.get("ratcheted") else "Stop-Loss Hit"
    except Exception as e:
        log.error("settle_closed_trade lookup failed: %s", e)
    finalize_trade(ctx, exit_px, reason)


def flatten(client: TradingClient, reason: str, reprotect: bool = True) -> str:
    with order_lock:
        ctx = store.get("trade_ctx")
        cancel_symbol_orders(client)
        pos, _ = get_position_details(client)
        if pos == 0:
            if ctx:
                settle_closed_trade(client, ctx, reason)
            return "Already flat"
        try:
            o = api_call(client.close_position, SYMBOL)
            filled = wait_fill(client, o.id)
        except APIError as e:
            notify(f"FLATTEN FAILED ({reason}): {e}")
            if reprotect and ctx:
                try:
                    place_oco(client, pos, ctx["tp"], ctx["sl"])
                except APIError as e2:
                    notify(f"CRITICAL: POSITION {pos:+d} UNPROTECTED: {e2}")
            return "Flatten failed"
        exit_px = (
            float(filled.filled_avg_price)
            if filled is not None and filled.filled_avg_price
            else None
        )
        if ctx:
            finalize_trade(ctx, exit_px, reason)
        return "Flattened"


def ensure_protection(client: TradingClient, ctx: dict, pos: int) -> tuple:
    for _ in range(3):
        tp_leg, sl_leg = find_exit_legs(client)
        if tp_leg and sl_leg:
            return tp_leg, sl_leg
        time.sleep(0.5)
    notify(f"Rebuilding missing OCO protection on {pos:+d} {SYMBOL}: TP={ctx['tp']} SL={ctx['sl']}")
    cancel_symbol_orders(client)
    try:
        place_oco(client, pos, ctx["tp"], ctx["sl"])
    except APIError as e:
        notify(f"Failed to place OCO ({e}); emergency flattening")
        flatten(client, "Protection build failed", reprotect=False)
        return None, None
    time.sleep(0.5)
    return find_exit_legs(client)


def adopt_position(client: TradingClient, pos: int, avg: float, sig: dict, now_ny: datetime) -> dict:
    d = 1 if pos > 0 else -1
    atr = sig["atr"]
    tp, sl = compute_brackets(avg, d, sig)
    tp_leg, sl_leg = find_exit_legs(client)
    if tp_leg and sl_leg:
        tp, sl = float(tp_leg.limit_price), float(sl_leg.stop_price)
    ctx = {
        "side": d,
        "qty": abs(pos),
        "entry_px": avg,
        "atr": atr,
        "regime": "ADOPTED",
        "tp": tp,
        "sl": sl,
        "ratcheted": (sl - avg) * d >= 0,
        "bar_id": None,
        "day": now_ny.strftime("%Y-%m-%d"),
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    ctx["trade_id"] = store.trade_open(ctx, sig)
    store.set("trade_ctx", ctx)
    notify(f"Adopted existing position {pos:+d} @ {avg:.2f} (TP={tp:.2f}, SL={sl:.2f})")
    return ctx


def _sync_dashboard(pos: int, ctx: dict | None, tp_leg, sl_leg):
    bot_state["position"] = pos
    if pos and ctx:
        bot_state["entry_price"] = round(ctx["entry_px"], 2)
        bot_state["active_tp"] = round(float(tp_leg.limit_price), 2) if tp_leg else round(ctx["tp"], 2)
        bot_state["active_sl"] = round(float(sl_leg.stop_price), 2) if sl_leg else round(ctx["sl"], 2)
        bot_state["ratchet_locked"] = bool(ctx.get("ratcheted"))
    else:
        bot_state.update(entry_price=0.0, active_tp=0.0, active_sl=0.0, ratchet_locked=False)
    until = float(store.get("last_exit_ts", 0.0)) + COOLDOWN_MINUTES * 60
    bot_state["cooldown_until"] = (
        datetime.fromtimestamp(until, ny_tz).strftime("%I:%M:%S %p %Z")
        if time.time() < until
        else "None"
    )


def reconcile(client: TradingClient, sig: dict, now_ny: datetime) -> tuple:
    with order_lock:
        pos, avg = get_position_details(client)
        ctx = store.get("trade_ctx")

        if ctx and (pos == 0 or (ctx["side"] > 0) != (pos > 0)):
            settle_closed_trade(client, ctx)
            ctx = None

        if pos == 0:
            if live_orders(client):
                cancel_symbol_orders(client)
            _sync_dashboard(0, None, None, None)
            return 0, None, None, None

        if ctx is None:
            ctx = adopt_position(client, pos, avg, sig, now_ny)

        tp_leg, sl_leg = ensure_protection(client, ctx, pos)
        if tp_leg is None and sl_leg is None and store.get("trade_ctx") is None:
            _sync_dashboard(0, None, None, None)
            return 0, None, None, None

        _sync_dashboard(pos, ctx, tp_leg, sl_leg)
        return pos, ctx, tp_leg, sl_leg


# =============================================================================
# 5. V9.1 QUANT ENGINE (TIMESTAMP-VERIFIED CLOSED BARS + STALENESS GUARD)
# =============================================================================
def compute_v9_signals(df: pd.DataFrame, prev_regime_trend: bool = False) -> dict:
    df = df.copy()
    latest_trade_px = df.attrs.get("latest_trade_price")
    now_utc = datetime.now(timezone.utc)

    df.index = df.index.tz_convert(ny_tz)
    if RTH_ONLY:
        rth_mask = (df.index.time >= dtime(9, 30)) & (df.index.time < dtime(16, 0))
        df_rth = df.loc[rth_mask].copy()
        if len(df_rth) >= 60:
            df = df_rth

    # Session-Anchored Intraday VWAP
    tp_bar = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp_bar * df["volume"]
    session_dates = df.index.date
    cum_pv = pv.groupby(session_dates).cumsum()
    cum_vol = df["volume"].replace(0, 1).groupby(session_dates).cumsum()
    df["vwap"] = cum_pv / cum_vol

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100.0 * (
        plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    )
    minus_di = 100.0 * (
        minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"].replace(0, 1e-9)
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    df["upper_band"] = df["ema21"] + (df["atr"] * MR_BAND_MULT)
    df["lower_band"] = df["ema21"] - (df["atr"] * MR_BAND_MULT)

    # Timestamp-Verified Closed-Bar Selection:
    # A 5m bar indexed at `t` covers `[t, t + 5m)`. If `t + 5m > now_utc`, `iloc[-1]` is still forming.
    last_bar_close_utc = df.index[-1].astimezone(timezone.utc) + timedelta(minutes=5)
    if last_bar_close_utc > now_utc:
        confirmed_df = df.iloc[:-1]
    else:
        confirmed_df = df

    c = confirmed_df.iloc[-1]
    prev = confirmed_df.iloc[-2]
    recent3 = confirmed_df.iloc[-3:]
    c_close_utc = confirmed_df.index[-1].astimezone(timezone.utc) + timedelta(minutes=5)
    bar_age_seconds = max(0.0, (now_utc - c_close_utc).total_seconds())
    bar_is_fresh = bar_age_seconds <= MAX_BAR_AGE_SECONDS

    live_price = float(latest_trade_px) if latest_trade_px else float(df.iloc[-1]["close"])
    closed_price = float(c["close"])
    atr_val = max(float(c["atr"]), 0.05)
    rsi_val = float(c["rsi"])
    adx_val = float(c["adx"])
    vwap_val = float(c["vwap"])
    ema21_val = float(c["ema21"])
    bar_time_str = str(confirmed_df.index[-1])

    if adx_val >= ADX_TREND_MIN:
        is_trend = True
    elif adx_val < ADX_RANGE_MAX:
        is_trend = False
    else:
        is_trend = prev_regime_trend

    bar_range = float(c["high"] - c["low"])
    vol_shock = bar_range > (atr_val * 2.5)

    c_range = max(bar_range, 0.01)
    bull_strength = (float(c["close"] - c["low"]) / c_range) >= 0.30
    bear_strength = (float(c["high"] - c["close"]) / c_range) >= 0.30

    was_oversold = (recent3["low"] <= recent3["lower_band"]).any() and (recent3["rsi"] < 40).any()
    was_overbought = (recent3["high"] >= recent3["upper_band"]).any() and (recent3["rsi"] > 60).any()

    rb_long = (
        bar_is_fresh
        and not vol_shock
        and was_oversold
        and c["close"] > c["open"]
        and c["close"] > prev["close"]
        and c["rsi"] > prev["rsi"]
        and bull_strength
    )
    rb_short = (
        bar_is_fresh
        and not vol_shock
        and was_overbought
        and c["close"] < c["open"]
        and c["close"] < prev["close"]
        and c["rsi"] < prev["rsi"]
        and bear_strength
        and c["close"] < vwap_val
    )

    ema_slope_up = ((c["ema21"] - confirmed_df.iloc[-4]["ema21"]) / (atr_val * 3.0)) >= 0.08
    ema_slope_dn = ((confirmed_df.iloc[-4]["ema21"] - c["ema21"]) / (atr_val * 3.0)) >= 0.08

    touched_pb_long = c["low"] <= (c["ema21"] + atr_val * 0.15) and c["low"] >= (
        c["ema50"] - atr_val * 0.25
    )
    touched_pb_short = c["high"] >= (c["ema21"] - atr_val * 0.15) and c["high"] <= (
        c["ema50"] + atr_val * 0.25
    )

    tb_long = (
        bar_is_fresh
        and not vol_shock
        and c["ema9"] > c["ema21"]
        and c["close"] > c["ema50"]
        and c["close"] > vwap_val
        and ema_slope_up
        and touched_pb_long
        and c["close"] > c["ema9"]
        and c["close"] > c["open"]
        and 45 < rsi_val < 72
    )
    tb_short = (
        bar_is_fresh
        and not vol_shock
        and c["ema9"] < c["ema21"]
        and c["close"] < c["ema50"]
        and c["close"] < vwap_val
        and ema_slope_dn
        and touched_pb_short
        and c["close"] < c["ema9"]
        and c["close"] < c["open"]
        and 28 < rsi_val < 55
    )

    long_sig = tb_long if is_trend else rb_long
    short_sig = tb_short if is_trend else rb_short

    return {
        "live_price": live_price,
        "closed_price": closed_price,
        "closed_bar_age_s": round(bar_age_seconds, 1),
        "atr": atr_val,
        "rsi": rsi_val,
        "adx": adx_val,
        "vwap": vwap_val,
        "ema21": ema21_val,
        "is_trend": is_trend,
        "long_sig": bool(long_sig),
        "short_sig": bool(short_sig),
        "bar_id": bar_time_str,
    }


# =============================================================================
# 6. EXECUTION, FILL RE-ANCHORING & RATCHET MANAGEMENT
# =============================================================================
def compute_brackets(entry_px: float, direction: int, sig: dict) -> tuple[float, float]:
    atr = sig["atr"]
    if sig["is_trend"]:
        tp = round(entry_px + direction * atr * TB_TP_MULT, 2)
        sl = round(entry_px - direction * atr * TB_SL_MULT, 2)
    else:
        min_tp_dist = atr * MR_MIN_TP_MULT
        forward_dists = [
            (anchor - entry_px) * direction
            for anchor in (sig["ema21"], sig["vwap"])
            if (anchor - entry_px) * direction > 0
        ]
        struct_dist = max(forward_dists) if forward_dists else min_tp_dist
        tp_dist = max(min_tp_dist, min(struct_dist, atr * 1.85))
        tp = round(entry_px + direction * tp_dist, 2)
        sl = round(entry_px - direction * atr * MR_SL_MULT, 2)
    return tp, sl


def open_trade(
    client: TradingClient, sig: dict, direction: int, label: str, manual: bool = False
) -> str:
    with order_lock:
        if get_position_details(client)[0] != 0:
            return "Not flat"
        cancel_symbol_orders(client)

        px = sig["live_price"]
        tp, sl = compute_brackets(px, direction, sig)

        tag = re.sub(r"[^0-9A-Za-z]", "", sig["bar_id"])[-16:]
        coid = f"spy9-{tag}-{'L' if direction > 0 else 'S'}{'M' if manual else 'A'}"
        opened_at = datetime.now(timezone.utc)
        store.set("last_bar_traded", sig["bar_id"])

        try:
            parent = api_call(
                client.submit_order,
                MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=QTY,
                    side=OrderSide.BUY if direction > 0 else OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=tp),
                    stop_loss=StopLossRequest(stop_price=sl),
                    client_order_id=coid,
                ),
            )
        except APIError as e:
            notify(f"Entry rejected ({label}): {e}")
            return "Entry rejected"

        filled = wait_fill(client, parent.id, ENTRY_FILL_TIMEOUT)
        if filled is None or _v(filled.status) != "filled":
            try:
                api_call(client.cancel_order_by_id, parent.id, retries=1)
            except APIError:
                pass

        pos, avg = get_position_details(client)
        if pos == 0:
            cancel_symbol_orders(client)
            return "No fill"

        actual_tp, actual_sl = compute_brackets(avg, direction, sig)
        tp_leg, sl_leg = find_exit_legs(client)
        if abs(avg - px) >= 0.05:
            if sl_leg:
                try:
                    api_call(
                        client.replace_order_by_id,
                        sl_leg.id,
                        ReplaceOrderRequest(stop_price=actual_sl),
                    )
                    sl = actual_sl
                except APIError as e:
                    log.warning("Could not re-anchor SL to fill price: %s", e)
            if tp_leg:
                try:
                    api_call(
                        client.replace_order_by_id,
                        tp_leg.id,
                        ReplaceOrderRequest(limit_price=actual_tp),
                    )
                    tp = actual_tp
                except APIError as e:
                    log.warning("Could not re-anchor TP to fill price: %s", e)

        ctx = {
            "side": direction,
            "qty": abs(pos),
            "entry_px": avg,
            "atr": sig["atr"],
            "regime": bot_state["regime"],
            "tp": tp,
            "sl": sl,
            "ratcheted": False,
            "bar_id": sig["bar_id"],
            "day": datetime.now(ny_tz).strftime("%Y-%m-%d"),
            "opened_at": opened_at.isoformat(),
        }
        ctx["trade_id"] = store.trade_open(ctx, sig)
        store.set("trade_ctx", ctx)
        notify(f"ENTRY {label}: {pos:+d} @ {avg:.2f} | TP={tp} | SL={sl} (sig_px={px:.2f})")
        ensure_protection(client, ctx, pos)
        bot_state["last_action"] = f"{label} {pos:+d} @ {avg:.2f}"
        return "Entered"


def manage_open_position(
    client: TradingClient, pos: int, ctx: dict, tp_leg, sl_leg, live_price: float
):
    try:
        opened_dt = datetime.fromisoformat(ctx["opened_at"])
        held_mins = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 60.0
        if held_mins >= MAX_HOLD_MINUTES and not ctx.get("ratcheted"):
            flatten(client, f"Time-Stop ({MAX_HOLD_MINUTES}m)")
            return
    except Exception as e:
        log.warning("Time-stop check error: %s", e)

    if ctx.get("ratcheted") or not (tp_leg and sl_leg):
        return
    d = 1 if pos > 0 else -1
    entry = ctx["entry_px"]
    dist = abs(float(tp_leg.limit_price) - entry)
    if (live_price - entry) * d < dist * RATCHET_TRIG:
        return

    new_sl = round(entry + d * ctx["atr"] * RATCHET_LOCK, 2)
    if (new_sl - float(sl_leg.stop_price)) * d <= 0:
        ctx["ratcheted"] = True
        store.set("trade_ctx", ctx)
        return

    try:
        api_call(
            client.replace_order_by_id,
            sl_leg.id,
            ReplaceOrderRequest(stop_price=new_sl),
        )
    except APIError as e:
        log.warning("Ratchet replace_order failed (will retry next scan): %s", e)
        return

    ctx["ratcheted"], ctx["sl"] = True, new_sl
    store.set("trade_ctx", ctx)
    notify(f"Ratchet Locked: Stop moved to {new_sl:.2f} (+{RATCHET_LOCK}x ATR)")


def risk_gate(client: TradingClient, now_ny: datetime) -> tuple[bool, str]:
    if store.get("kill_switch", False):
        return False, "Kill switch active"
    acct = api_call(client.get_account)
    day_pnl = float(acct.equity) - float(acct.last_equity)
    bot_state["day_pnl"] = round(day_pnl, 2)
    if day_pnl <= -MAX_DAILY_LOSS_USD:
        return False, f"Daily loss cap hit (${day_pnl:.2f})"
    n, _ = broker_session_metrics(client, now_ny)
    if n >= MAX_TRADES_PER_DAY:
        return False, f"Max trades/day reached ({n}/{MAX_TRADES_PER_DAY})"
    return True, ""


# =============================================================================
# 7. SELF-HEALING AUTONOMOUS SCANNER LOOP
# =============================================================================
def _scan():
    now_ny = datetime.now(ny_tz)
    bot_state["last_check"] = now_ny.strftime("%Y-%m-%d %I:%M:%S %p %Z")
    bot_state["scan_step"] = "1. Checking API Keys"

    api_key, secret_key, _ = get_env_keys()
    if not (api_key and secret_key):
        bot_state["regime"] = "Waiting for ALPACA_API_KEY & ALPACA_SECRET_KEY"
        return

    trade_client, data_client = get_clients()
    clock = api_call(trade_client.get_clock)
    is_open = bool(clock.is_open)
    mins_to_close = (
        (clock.next_close - clock.timestamp).total_seconds() / 60.0 if is_open else None
    )
    hhmm = now_ny.hour * 100 + now_ny.minute

    in_entry_window = (
        is_open
        and hhmm >= 940
        and mins_to_close is not None
        and mins_to_close >= NO_ENTRY_LAST_MIN
    )
    in_eod_flatten = (
        is_open and mins_to_close is not None and mins_to_close <= EOD_FLATTEN_MIN
    )
    bot_state["market_open"] = is_open
    bot_state["active_window"] = (
        "ACTIVE ENTRY WINDOW"
        if in_entry_window
        else ("EOD FLATTEN" if in_eod_flatten else "OUTSIDE ENTRY HOURS")
    )

    bot_state["scan_step"] = "2. Downloading SPY 5m Bars"
    bars = fetch_bars_with_fallback(data_client)
    if len(bars) < 60:
        bot_state["scan_step"] = f"Waiting for sufficient bars ({len(bars)}/60)"
        return

    bot_state["scan_step"] = "3. Computing v9.1 Signals"
    prev_trend = bool(store.get("prev_regime_trend", False))
    sig = compute_v9_signals(bars, prev_regime_trend=prev_trend)
    store.set("prev_regime_trend", sig["is_trend"])

    bot_state.update(
        spy_price=round(sig["live_price"], 2),
        last_5m_close=round(sig["closed_price"], 2),
        closed_bar_age_s=sig["closed_bar_age_s"],
        session_vwap=round(sig["vwap"], 2),
        rsi=round(sig["rsi"], 1),
        adx=round(sig["adx"], 1),
        atr=round(sig["atr"], 2),
        regime="TREND (EMA+VWAP Bounce)" if sig["is_trend"] else "RANGE (Rubberband MR)",
    )

    bot_state["scan_step"] = "4. Reconciling Broker State"
    pos, ctx, tp_leg, sl_leg = reconcile(trade_client, sig, now_ny)
    if not is_open:
        bot_state["scan_step"] = "Complete (Market Closed)"
        return

    ok, why = risk_gate(trade_client, now_ny)
    bot_state["risk_status"] = "OK" if ok else why

    if pos != 0:
        if in_eod_flatten:
            flatten(trade_client, "EOD Flatten")
        elif not ok:
            flatten(trade_client, f"Risk Halt: {why}")
        else:
            manage_open_position(trade_client, pos, ctx, tp_leg, sl_leg, sig["live_price"])
        bot_state["scan_step"] = "Complete (Position Managed)"
        return

    if in_entry_window and ok:
        bot_state["scan_step"] = "5. Evaluating Entries"
        last_exit_ts = float(store.get("last_exit_ts", 0.0))
        in_cooldown = (time.time() - last_exit_ts) < (COOLDOWN_MINUTES * 60)
        bar_done = store.get("last_bar_traded") == sig["bar_id"]
        pending = store.get("pending_entry")
        direction, label, manual = 0, "", False

        if pending and (time.time() - pending["ts"]) <= PENDING_TTL:
            direction = 1 if pending["signal"] == "BUY_LONG" else -1
            label, manual = f"MANUAL {pending['signal']}", True
        elif not in_cooldown and not bar_done:
            direction = 1 if sig["long_sig"] else (-1 if sig["short_sig"] else 0)
            label = f"{'LONG' if direction > 0 else 'SHORT'} {bot_state['regime']}"

        if pending:
            store.set("pending_entry", None)
        if direction:
            open_trade(trade_client, sig, direction, label, manual)

    bot_state["scan_step"] = "Complete (All Data Synced)"


def run_single_brain_scan():
    global scan_errors
    if not scan_lock.acquire(blocking=False):
        return
    try:
        _scan()
        if scan_errors >= 3:
            notify("Brain scans recovered cleanly.")
        scan_errors = 0
    except APIError as e:
        scan_errors += 1
        log.error("Alpaca APIError during scan: %s", e)
        bot_state["scan_step"] = "Failed with APIError"
        bot_state["last_action"] = f"APIError: {str(e)[:120]}"
        if scan_errors == 3:
            notify(f"3 consecutive APIErrors: {str(e)[:180]}. Broker OCO legs remain active.")
    except Exception as e:
        scan_errors += 1
        log.exception("Unexpected scan error: %s", e)
        bot_state["scan_step"] = "Failed with Error"
        bot_state["last_action"] = f"Error: {str(e)[:120]}"
        if scan_errors == 3:
            notify(f"3 consecutive scan errors: {str(e)[:180]}. Broker OCO legs remain active.")
    finally:
        scan_lock.release()


def background_loop():
    while True:
        try:
            leader = store.acquire_lease(INSTANCE_ID, LEASE_TTL)
            bot_state["leader"] = leader
            if leader:
                run_single_brain_scan()
        except Exception as e:
            log.exception("Background loop error: %s", e)
        time.sleep(CHECK_SECONDS)


def start_worker_once():
    global worker_started
    if not worker_started:
        worker_started = True
        threading.Thread(target=background_loop, daemon=True, name="brain-v9.1-loop").start()


# =============================================================================
# 8. AUTHENTICATED HTTP & WEBHOOK ENDPOINTS
# =============================================================================
def _token_ok() -> bool:
    token = os.environ.get("STATUS_TOKEN", "").strip()
    supplied = request.headers.get("X-Status-Token", "").strip()
    return bool(token) and hmac.compare_digest(supplied.encode(), token.encode())


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "alive", "version": bot_state["version"], "leader": bot_state["leader"]}), 200


@app.route("/status", methods=["GET"])
def status():
    if not _token_ok():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(
        {
            "dashboard": bot_state,
            "trade_ctx": store.get("trade_ctx"),
            "kill_switch": store.get("kill_switch", False),
        }
    ), 200


@app.route("/trades", methods=["GET"])
def trades():
    if not _token_ok():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(store.recent_trades(200)), 200


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True, silent=True) or {}
    secret = os.environ.get("WEBHOOK_PASSPHRASE", "").strip()
    if not secret:
        return jsonify({"error": "Webhook disabled (unconfigured secret)"}), 503
    if not hmac.compare_digest(str(data.get("passphrase", "")).encode(), secret.encode()):
        return jsonify({"error": "Unauthorized"}), 401

    sig = data.get("signal")
    try:
        if sig in ("BUY_LONG", "BUY_SHORT"):
            store.set("pending_entry", {"signal": sig, "ts": time.time()})
            return jsonify({"status": "Queued for bracketed risk-gated entry on next scan"}), 200
        if sig == "EXIT_POSITION":
            return jsonify({"status": flatten(get_clients()[0], "Manual Webhook EXIT")}), 200
        if sig == "KILL":
            store.set("kill_switch", True)
            return jsonify({"status": flatten(get_clients()[0], "KILL Switch Activated")}), 200
        if sig == "RESUME":
            store.set("kill_switch", False)
            return jsonify({"status": "Kill switch cleared; scanning resumed"}), 200
    except APIError as e:
        log.error("Alpaca APIError in webhook: %s", e)
        return jsonify({"error": f"Alpaca APIError: {str(e)}"}), 502

    return jsonify({"error": "Unknown signal"}), 400


start_worker_once()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
