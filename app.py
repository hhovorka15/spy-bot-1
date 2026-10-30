import hmac
import logging
import os
import re
import threading
import time
from datetime import datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
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
log = logging.getLogger("spy-brain")

# ==========================================
# 1. MASTER CONFIGURATION (v9.2 STATELESS)
# ==========================================
SYMBOL = os.environ.get("TRADE_SYMBOL", "SPY").strip().upper()
QTY = int(os.environ.get("TRADE_QTY", "10"))
CHECK_SECONDS = int(os.environ.get("CHECK_SECONDS", "30"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "15"))
MAX_HOLD_MINUTES = int(os.environ.get("MAX_HOLD_MINUTES", "90"))
MAX_BAR_AGE_SECONDS = int(os.environ.get("MAX_BAR_AGE_SECONDS", "360"))

# Quantitative Parameters
ADX_RANGE_MAX = 20.0   # < 20 = Confirmed Range (Rubberband MR)
ADX_TREND_MIN = 25.0   # >= 25 = Confirmed Trend (EMA + VWAP Bounce)
MR_BAND_MULT = 1.60
MR_MIN_TP_MULT = 1.20
MR_SL_MULT = 1.50
TB_TP_MULT = 2.25
TB_SL_MULT = 1.25
RATCHET_TRIG = 0.75    # Locks profit at 75% of TP distance
RATCHET_LOCK = 0.35    # Locks +0.35x ATR beyond entry

# Account Risk & Session Controls
MAX_DAILY_LOSS_USD = float(os.environ.get("MAX_DAILY_LOSS_USD", "150.0"))
MAX_TRADES_PER_DAY = int(os.environ.get("MAX_TRADES_PER_DAY", "6"))
EOD_FLATTEN_MIN = 5
NO_ENTRY_LAST_MIN = 20
ENTRY_FILL_TIMEOUT = 10.0
PREFERRED_FEED = os.environ.get("ALPACA_DATA_FEED", "SIP").strip().upper()

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

app = Flask(__name__)
order_lock = threading.RLock()
brain_lock = threading.Lock()
ny_tz = ZoneInfo("America/New_York")

bot_state = {
    "last_check": "Not checked yet",
    "market_open": False,
    "active_window": "Checking...",
    "data_feed": PREFERRED_FEED,
    "spy_price": 0.0,
    "last_5m_close": 0.0,
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
    "last_bar_traded": "",
    "cooldown_until": "None",
    "trades_today": 0,
    "day_pnl": 0.0,
    "risk_status": "OK",
    "last_action": "System booted cleanly",
}

# In-memory flags (reconstructed from Alpaca order history on every scan)
last_exit_ts = 0.0
prev_regime_trend = False
kill_switch = False
pending_webhook = None
_clients = None
worker_started = False


# ==========================================
# 2. ALPACA CLIENT & RATE-LIMIT WRAPPERS
# ==========================================
def _v(x) -> str:
    return str(getattr(x, "value", x)).lower()


def get_env_keys() -> tuple[str, str, bool]:
    api_key = (
        os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_KEY") or ""
    ).strip()
    secret_key = (
        os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET") or ""
    ).strip()
    paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
    return api_key, secret_key, paper


def get_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    global _clients
    if _clients is None:
        api_key, secret_key, paper = get_env_keys()
        if not api_key or not secret_key:
            raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_KEY or ALPACA_SECRET_KEY/ALPACA_SECRET")
        _clients = (
            TradingClient(api_key, secret_key, paper=paper),
            StockHistoricalDataClient(api_key, secret_key),
        )
    return _clients


def api_call(fn, *args, retries: int = 3, **kwargs):
    """Wraps every Alpaca SDK call in APIError handling with exponential backoff."""
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
    """Returns (signed_qty, avg_entry_price)."""
    try:
        pos = api_call(client.get_open_position, SYMBOL)
        qty = abs(int(float(pos.qty)))
        signed_qty = -qty if pos.side == PositionSide.SHORT else qty
        return signed_qty, float(pos.avg_entry_price)
    except APIError as e:
        if getattr(e, "status_code", None) == 404 or "position does not exist" in str(e).lower():
            return 0, 0.0
        raise


def fetch_bars(data_client: StockHistoricalDataClient) -> pd.DataFrame:
    """Fetches 5m bars using SIP (with automatic fallback to IEX) plus latest real-time trade price."""
    start_time = datetime.now(timezone.utc) - timedelta(days=7)
    feed = DataFeed.SIP if PREFERRED_FEED == "SIP" else DataFeed.IEX
    try:
        req = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_time,
            feed=feed,
        )
        bars = api_call(data_client.get_stock_bars, req).df
        bot_state["data_feed"] = feed.value.upper()
    except APIError as e:
        if feed == DataFeed.SIP:
            log.warning("SIP feed unavailable (%s); falling back to IEX", e)
            req = StockBarsRequest(
                symbol_or_symbols=SYMBOL,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start_time,
                feed=DataFeed.IEX,
            )
            bars = api_call(data_client.get_stock_bars, req).df
            bot_state["data_feed"] = "IEX"
        else:
            raise

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(SYMBOL, level=0)
    bars = bars.sort_index()

    try:
        active_feed = DataFeed.IEX if bot_state["data_feed"] == "IEX" else DataFeed.SIP
        t_req = StockLatestTradeRequest(symbol_or_symbols=SYMBOL, feed=active_feed)
        latest = api_call(data_client.get_stock_latest_trade, t_req)
        if SYMBOL in latest and latest[SYMBOL].price:
            bars.attrs["latest_trade_price"] = float(latest[SYMBOL].price)
    except APIError as e:
        log.debug("Latest trade lookup skipped: %s", e)

    return bars


# ==========================================
# 3. STATELESS BROKER RECONCILIATION & OCO
# ==========================================
def get_recent_orders(client: TradingClient, after: datetime | None = None) -> list:
    after = after or (datetime.now(timezone.utc) - timedelta(days=2))
    req = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        symbols=[SYMBOL],
        after=after,
        nested=True,
        limit=100,
        direction=Sort.DESC,
    )
    return api_call(client.get_orders, req)


def get_live_orders(client: TradingClient, orders: list | None = None) -> list:
    orders = orders if orders is not None else get_recent_orders(client)
    out = []
    for o in orders:
        for x in [o, *(getattr(o, "legs", None) or [])]:
            if _v(x.status) in LIVE_STATUSES:
                out.append(x)
    return out


def find_exit_legs(client: TradingClient, orders: list | None = None) -> tuple:
    tp_leg = sl_leg = None
    for o in get_live_orders(client, orders):
        t = _v(o.order_type)
        if t == "limit" and tp_leg is None:
            tp_leg = o
        elif t in ("stop", "stop_limit") and sl_leg is None:
            sl_leg = o
    return tp_leg, sl_leg


def cancel_symbol_orders(client: TradingClient, timeout: float = 6.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        live = get_live_orders(client)
        if not live:
            return True
        for o in live:
            try:
                api_call(client.cancel_order_by_id, o.id, retries=1)
            except APIError:
                pass
        time.sleep(0.4)
    return not get_live_orders(client)


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


def compute_brackets(entry_px: float, direction: int, sig: dict) -> tuple[float, float]:
    """Calculates dynamic ATR brackets: 1.80 R:R for Trend; EMA21/VWAP structural anchor for MR."""
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


def flatten_position(client: TradingClient, reason: str) -> str:
    """Cancels resting OCO legs and closes any open SPY position at market."""
    global last_exit_ts
    with order_lock:
        cancel_symbol_orders(client)
        pos, _ = get_position_details(client)
        if pos == 0:
            bot_state.update(
                position=0, entry_price=0.0, active_tp=0.0, active_sl=0.0, ratchet_locked=False
            )
            return "Already flat"

        log.info("Flattening position (%+d) | Reason: %s", pos, reason)
        try:
            o = api_call(client.close_position, SYMBOL)
            wait_fill(client, o.id)
        except APIError as e:
            log.error("APIError flattening position (%s): %s", reason, e)
            if bot_state["active_tp"] > 0 and bot_state["active_sl"] > 0:
                try:
                    place_oco(client, pos, bot_state["active_tp"], bot_state["active_sl"])
                except APIError as e2:
                    log.error("CRITICAL: Failed to restore OCO after flatten error: %s", e2)
            raise

        last_exit_ts = time.time()
        cooldown_dt = datetime.fromtimestamp(last_exit_ts + COOLDOWN_MINUTES * 60, ny_tz)
        bot_state.update(
            position=0,
            entry_price=0.0,
            active_tp=0.0,
            active_sl=0.0,
            ratchet_locked=False,
            cooldown_until=cooldown_dt.strftime("%I:%M:%S %p %Z"),
            last_action=f"Exited ({reason})",
        )
        return "Flattened"


def sync_broker_state(client: TradingClient, sig: dict, now_ny: datetime) -> tuple:
    """
    Statelessly reconstructs position, brackets, ratchet lock, entry timestamp,
    today's trade count, last traded bar, and exit cooldown directly from Alpaca.
    """
    global last_exit_ts
    with order_lock:
        session_start = datetime.combine(now_ny.date(), dtime(0, 0), tzinfo=ny_tz).astimezone(timezone.utc)
        orders = get_recent_orders(client, after=session_start - timedelta(hours=12))
        pos, avg_entry = get_position_details(client)

        entries_today = 0
        latest_exit_fill_ts = last_exit_ts
        latest_entry_dt = None

        for o in orders:
            coid = str(getattr(o, "client_order_id", "") or "")
            if o.filled_at and o.filled_at >= session_start:
                if coid.startswith("spy-"):
                    entries_today += 1
                    m = re.match(r"^spy-(\d{12})-", coid)
                    if m and not bot_state["last_bar_traded"]:
                        bot_state["last_bar_traded"] = m.group(1)

            if pos != 0:
                entry_side = "buy" if pos > 0 else "sell"
                if _v(o.side) == entry_side and o.filled_at:
                    if latest_entry_dt is None or o.filled_at > latest_entry_dt:
                        latest_entry_dt = o.filled_at

            for leg in getattr(o, "legs", None) or []:
                if leg.filled_at and float(leg.filled_qty or 0) > 0:
                    ts = leg.filled_at.timestamp()
                    if ts > latest_exit_fill_ts:
                        latest_exit_fill_ts = ts
                        if pos == 0 and bot_state["position"] != 0:
                            leg_type = _v(leg.order_type)
                            exit_label = (
                                "Take-Profit Hit"
                                if leg_type == "limit"
                                else ("Ratchet Lock Hit" if bot_state["ratchet_locked"] else "Stop-Loss Hit")
                            )
                            bot_state["last_action"] = f"Exited ({exit_label})"
                            log.info("Detected broker-side exit: %s @ %s", exit_label, leg.filled_avg_price)

        last_exit_ts = max(last_exit_ts, latest_exit_fill_ts)
        bot_state["trades_today"] = entries_today
        until = last_exit_ts + COOLDOWN_MINUTES * 60
        bot_state["cooldown_until"] = (
            datetime.fromtimestamp(until, ny_tz).strftime("%I:%M:%S %p %Z")
            if time.time() < until
            else "None"
        )

        if pos == 0:
            if get_live_orders(client, orders):
                cancel_symbol_orders(client)
            bot_state.update(
                position=0, entry_price=0.0, active_tp=0.0, active_sl=0.0, ratchet_locked=False
            )
            return 0, None, None, None

        direction = 1 if pos > 0 else -1
        tp_leg, sl_leg = find_exit_legs(client, orders)

        # Self-heal missing broker OCO brackets if restarted or adopted mid-trade
        if not (tp_leg and sl_leg):
            for _ in range(2):
                time.sleep(0.4)
                tp_leg, sl_leg = find_exit_legs(client)
                if tp_leg and sl_leg:
                    break
            if not (tp_leg and sl_leg):
                tp, sl = compute_brackets(avg_entry, direction, sig)
                log.warning("Rebuilding missing OCO legs for %+d %s: TP=%s SL=%s", pos, SYMBOL, tp, sl)
                cancel_symbol_orders(client)
                try:
                    place_oco(client, pos, tp, sl)
                    time.sleep(0.5)
                    tp_leg, sl_leg = find_exit_legs(client)
                except APIError as e:
                    log.error("Failed to rebuild OCO (%s); emergency flattening", e)
                    flatten_position(client, "OCO Rebuild Failed")
                    return 0, None, None, None

        active_tp = round(float(tp_leg.limit_price), 2) if tp_leg else 0.0
        active_sl = round(float(sl_leg.stop_price), 2) if sl_leg else 0.0
        is_ratcheted = (active_sl - avg_entry) * direction > 0 if active_sl else False

        bot_state.update(
            position=pos,
            entry_price=round(avg_entry, 2),
            active_tp=active_tp,
            active_sl=active_sl,
            ratchet_locked=is_ratcheted,
        )
        return pos, tp_leg, sl_leg, latest_entry_dt


def enter_bracket_trade(
    client: TradingClient, sig: dict, direction: int, label: str, manual: bool = False
) -> str:
    """Submits an atomic broker-side BRACKET order and re-anchors legs on fill slippage."""
    with order_lock:
        if get_position_details(client)[0] != 0:
            return "Already in position"
        cancel_symbol_orders(client)

        px = sig["live_price"]
        tp, sl = compute_brackets(px, direction, sig)
        bar_tag = sig["bar_tag"]
        coid = f"spy-{bar_tag}-{'L' if direction > 0 else 'S'}{'M' if manual else 'A'}"
        bot_state["last_bar_traded"] = bar_tag

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
            log.error("Bracket entry rejected (%s): %s", label, e)
            return "Entry rejected"

        filled = wait_fill(client, parent.id, ENTRY_FILL_TIMEOUT)
        if filled is None or _v(filled.status) != "filled":
            try:
                api_call(client.cancel_order_by_id, parent.id, retries=1)
            except APIError:
                pass

        pos, avg_entry = get_position_details(client)
        if pos == 0:
            cancel_symbol_orders(client)
            return "No fill"

        actual_tp, actual_sl = compute_brackets(avg_entry, direction, sig)
        tp_leg, sl_leg = find_exit_legs(client)
        if abs(avg_entry - px) >= 0.05:
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

        bot_state.update(
            position=pos,
            entry_price=round(avg_entry, 2),
            active_tp=tp,
            active_sl=sl,
            ratchet_locked=False,
            last_action=f"{label} {pos:+d} @ {avg_entry:.2f}",
        )
        log.info("ENTERED %s: %+d %s @ %.2f | TP=%.2f | SL=%.2f", label, pos, SYMBOL, avg_entry, tp, sl)
        return "Order sent"


def manage_open_position(
    client: TradingClient,
    pos: int,
    tp_leg,
    sl_leg,
    entry_dt: datetime | None,
    live_price: float,
    atr: float,
):
    """Enforces holding time-stop and updates the broker-side stop leg when ratchet triggers."""
    if entry_dt is not None and not bot_state["ratchet_locked"]:
        held_mins = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60.0
        if held_mins >= MAX_HOLD_MINUTES:
            flatten_position(client, f"Time-Stop ({MAX_HOLD_MINUTES}m)")
            return

    if bot_state["ratchet_locked"] or not (tp_leg and sl_leg):
        return

    direction = 1 if pos > 0 else -1
    entry_px = bot_state["entry_price"]
    target_dist = abs(float(tp_leg.limit_price) - entry_px)
    if (live_price - entry_px) * direction < target_dist * RATCHET_TRIG:
        return

    new_sl = round(entry_px + direction * atr * RATCHET_LOCK, 2)
    if (new_sl - float(sl_leg.stop_price)) * direction <= 0:
        bot_state["ratchet_locked"] = True
        return

    try:
        api_call(
            client.replace_order_by_id,
            sl_leg.id,
            ReplaceOrderRequest(stop_price=new_sl),
        )
        bot_state["active_sl"] = new_sl
        bot_state["ratchet_locked"] = True
        bot_state["last_action"] = f"Ratchet Locked SL @ {new_sl:.2f}"
        log.info("Ratchet triggered: Broker SL moved to %.2f", new_sl)
    except APIError as e:
        log.warning("Ratchet replace_order failed (will retry next scan): %s", e)


def check_risk_gate(client: TradingClient) -> tuple[bool, str]:
    if kill_switch:
        return False, "Kill switch active"
    acct = api_call(client.get_account)
    day_pnl = float(acct.equity) - float(acct.last_equity)
    bot_state["day_pnl"] = round(day_pnl, 2)
    if day_pnl <= -MAX_DAILY_LOSS_USD:
        return False, f"Daily loss cap hit (${day_pnl:.2f})"
    if bot_state["trades_today"] >= MAX_TRADES_PER_DAY:
        return False, f"Max trades/day ({bot_state['trades_today']}/{MAX_TRADES_PER_DAY})"
    return True, "OK"


# ==========================================
# 4. V9.2 CONFIRMED-BAR QUANT ENGINE
# ==========================================
def compute_signals(df: pd.DataFrame, prev_trend: bool = False) -> dict:
    df = df.copy()
    latest_trade_px = df.attrs.get("latest_trade_price")
    now_utc = datetime.now(timezone.utc)

    # Filter strictly to Regular Trading Hours (09:30 - 16:00 ET) so overnight gaps don't skew ATR/EMA
    df.index = df.index.tz_convert(ny_tz)
    rth_mask = (df.index.time >= dtime(9, 30)) & (df.index.time < dtime(16, 0))
    df_rth = df.loc[rth_mask].copy()
    if len(df_rth) >= 60:
        df = df_rth

    # Session-Anchored Intraday VWAP (resets daily at 09:30 ET)
    tp_bar = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp_bar * df["volume"]
    session_dates = df.index.date
    df["vwap"] = pv.groupby(session_dates).cumsum() / df["volume"].replace(0, 1).groupby(session_dates).cumsum()

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Wilder's ATR(14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # Wilder's RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder's ADX(14)
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100.0 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"].replace(0, 1e-9))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / df["atr"].replace(0, 1e-9))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # Keltner Stretch Bands
    df["upper_band"] = df["ema21"] + (df["atr"] * MR_BAND_MULT)
    df["lower_band"] = df["ema21"] - (df["atr"] * MR_BAND_MULT)

    # Timestamp-Verified Closed-Bar Selection
    last_bar_close_utc = df.index[-1].astimezone(timezone.utc) + timedelta(minutes=5)
    confirmed_df = df.iloc[:-1] if last_bar_close_utc > now_utc else df

    c = confirmed_df.iloc[-1]
    prev = confirmed_df.iloc[-2]
    recent3 = confirmed_df.iloc[-3:]

    c_close_utc = confirmed_df.index[-1].astimezone(timezone.utc) + timedelta(minutes=5)
    bar_is_fresh = max(0.0, (now_utc - c_close_utc).total_seconds()) <= MAX_BAR_AGE_SECONDS

    live_price = float(latest_trade_px) if latest_trade_px else float(df.iloc[-1]["close"])
    closed_price = float(c["close"])
    atr_val = max(float(c["atr"]), 0.05)
    rsi_val = float(c["rsi"])
    adx_val = float(c["adx"])
    vwap_val = float(c["vwap"])
    ema21_val = float(c["ema21"])
    bar_tag = confirmed_df.index[-1].strftime("%Y%m%d%H%M")

    # ADX Hysteresis Deadband
    if adx_val >= ADX_TREND_MIN:
        is_trend = True
    elif adx_val < ADX_RANGE_MAX:
        is_trend = False
    else:
        is_trend = prev_trend

    bar_range = float(c["high"] - c["low"])
    vol_shock = bar_range > (atr_val * 2.5)
    c_range = max(bar_range, 0.01)
    bull_strength = (float(c["close"] - c["low"]) / c_range) >= 0.30
    bear_strength = (float(c["high"] - c["close"]) / c_range) >= 0.30

    # Engine 1: 2-Step Rubberband Mean Reversion
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

    # Engine 2: EMA + Session VWAP Trend Bounce
    ema_slope_up = ((c["ema21"] - confirmed_df.iloc[-4]["ema21"]) / (atr_val * 3.0)) >= 0.08
    ema_slope_dn = ((confirmed_df.iloc[-4]["ema21"] - c["ema21"]) / (atr_val * 3.0)) >= 0.08

    touched_pb_long = c["low"] <= (c["ema21"] + atr_val * 0.15) and c["low"] >= (c["ema50"] - atr_val * 0.25)
    touched_pb_short = c["high"] >= (c["ema21"] - atr_val * 0.15) and c["high"] <= (c["ema50"] + atr_val * 0.25)

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
        "atr": atr_val,
        "rsi": rsi_val,
        "adx": adx_val,
        "vwap": vwap_val,
        "ema21": ema21_val,
        "is_trend": is_trend,
        "long_sig": bool(long_sig),
        "short_sig": bool(short_sig),
        "bar_tag": bar_tag,
    }


# ==========================================
# 5. SELF-HEALING AUTONOMOUS SCANNER
# ==========================================
def run_single_brain_scan():
    global prev_regime_trend, pending_webhook
    if not brain_lock.acquire(blocking=False):
        return
    try:
        now_ny = datetime.now(ny_tz)
        bot_state["last_check"] = now_ny.strftime("%Y-%m-%d %I:%M:%S %p %Z")

        api_key, secret_key, _ = get_env_keys()
        if not api_key or not secret_key:
            bot_state["regime"] = "Waiting for ALPACA_API_KEY & ALPACA_SECRET_KEY"
            return

        trade_client, data_client = get_clients()
        clock = api_call(trade_client.get_clock)
        is_open = bool(clock.is_open)
        mins_to_close = (
            (clock.next_close - clock.timestamp).total_seconds() / 60.0 if is_open else None
        )
        hhmm = now_ny.hour * 100 + now_ny.minute

        # Dynamic close handling (automatically handles 13:00 EST half-days via clock.next_close)
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

        bars = fetch_bars(data_client)
        if len(bars) < 60:
            return

        sig = compute_signals(bars, prev_trend=prev_regime_trend)
        prev_regime_trend = sig["is_trend"]

        bot_state.update(
            spy_price=round(sig["live_price"], 2),
            last_5m_close=round(sig["closed_price"], 2),
            session_vwap=round(sig["vwap"], 2),
            rsi=round(sig["rsi"], 1),
            adx=round(sig["adx"], 1),
            atr=round(sig["atr"], 2),
            regime="TREND (EMA+VWAP Bounce)" if sig["is_trend"] else "RANGE (Rubberband MR)",
        )

        pos, tp_leg, sl_leg, entry_dt = sync_broker_state(trade_client, sig, now_ny)
        if not is_open:
            return

        ok, risk_msg = check_risk_gate(trade_client)
        bot_state["risk_status"] = risk_msg

        if pos != 0:
            if in_eod_flatten:
                flatten_position(trade_client, "EOD Flatten")
            elif not ok:
                flatten_position(trade_client, f"Risk Halt ({risk_msg})")
            else:
                manage_open_position(
                    trade_client, pos, tp_leg, sl_leg, entry_dt, sig["live_price"], sig["atr"]
                )
            return

        if in_entry_window and ok:
            in_cooldown = (time.time() - last_exit_ts) < (COOLDOWN_MINUTES * 60)
            already_traded_bar = bot_state["last_bar_traded"] == sig["bar_tag"]

            direction, label, manual = 0, "", False
            if pending_webhook and (time.time() - pending_webhook["ts"]) <= 120:
                direction = 1 if pending_webhook["signal"] == "BUY_LONG" else -1
                label, manual = f"Manual Webhook {pending_webhook['signal']}", True
                pending_webhook = None
            elif not in_cooldown and not already_traded_bar:
                if sig["long_sig"]:
                    direction, label = 1, f"LONG {bot_state['regime']}"
                elif sig["short_sig"]:
                    direction, label = -1, f"SHORT {bot_state['regime']}"

            if direction != 0:
                enter_bracket_trade(trade_client, sig, direction, label, manual=manual)

    except APIError as e:
        log.error("Alpaca APIError in brain scan: %s", e)
        bot_state["last_action"] = f"APIError: {str(e)[:120]}"
    except Exception as e:
        log.exception("Brain scan error: %s", e)
        bot_state["last_action"] = f"Error: {str(e)[:120]}"
    finally:
        brain_lock.release()


def background_loop():
    while True:
        run_single_brain_scan()
        time.sleep(CHECK_SECONDS)


def start_worker_once():
    global worker_started
    if not worker_started:
        worker_started = True
        threading.Thread(target=background_loop, daemon=True, name="spy-brain-loop").start()


start_worker_once()


# ==========================================
# 6. HTTP & WEBHOOK ENDPOINTS
# ==========================================
@app.route("/", methods=["GET"])
def home():
    status_token = os.environ.get("STATUS_TOKEN", "").strip()
    if status_token:
        supplied = request.headers.get("X-Status-Token", "").strip()
        if not hmac.compare_digest(supplied.encode(), status_token.encode()):
            return jsonify({"status": "alive", "authenticated": False}), 200

    api_key, secret_key, _ = get_env_keys()
    return jsonify(
        {
            "status": "SPY Autonomous Brain v9.2 (Stateless Production) is Alive!",
            "keys_configured": bool(api_key and secret_key),
            "live_dashboard": bot_state,
        }
    ), 200


@app.route("/webhook", methods=["POST"])
def webhook():
    global kill_switch, pending_webhook
    data = request.get_json(force=True, silent=True) or {}
    webhook_pass = os.environ.get("WEBHOOK_PASSPHRASE", "").strip()
    if not webhook_pass:
        return jsonify({"error": "Webhook passphrase not configured on server"}), 503

    supplied = str(data.get("passphrase", ""))
    if not hmac.compare_digest(supplied.encode(), webhook_pass.encode()):
        return jsonify({"error": "Unauthorized"}), 401

    signal = data.get("signal")
    try:
        if signal in ("BUY_LONG", "BUY_SHORT"):
            pending_webhook = {"signal": signal, "ts": time.time()}
            return jsonify({"status": f"Queued {signal} for bracketed entry on next scan"}), 200
        if signal == "EXIT_POSITION":
            trade_client, _ = get_clients()
            return jsonify({"status": flatten_position(trade_client, "Manual Webhook EXIT")}), 200
        if signal == "KILL":
            kill_switch = True
            trade_client, _ = get_clients()
            return jsonify({"status": flatten_position(trade_client, "KILL Switch Activated")}), 200
        if signal == "RESUME":
            kill_switch = False
            return jsonify({"status": "Kill switch cleared; scanning resumed"}), 200
    except APIError as e:
        log.error("Alpaca APIError in webhook: %s", e)
        return jsonify({"error": f"Alpaca APIError: {str(e)}"}), 502

    return jsonify({"error": "Unknown signal"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
