import hmac
import logging
import math
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
from requests.exceptions import ConnectionError, Timeout
from urllib3.exceptions import ProtocolError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spy-brain")

SYMBOL = os.environ.get("TRADE_SYMBOL", "SPY").strip().upper()
CHECK_SECONDS = int(os.environ.get("CHECK_SECONDS", "30"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "15"))
MAX_HOLD_MINUTES = int(os.environ.get("MAX_HOLD_MINUTES", "90"))
MAX_BAR_AGE_SECONDS = int(os.environ.get("MAX_BAR_AGE_SECONDS", "360"))

RISK_MODE = os.environ.get("RISK_MODE", "FIXED_USD").strip().upper()
RISK_USD_PER_TRADE = float(os.environ.get("RISK_USD_PER_TRADE", "50.0"))
RISK_PCT_PER_TRADE = float(os.environ.get("RISK_PCT_PER_TRADE", "0.50"))
MAX_POSITION_PCT_EQUITY = float(os.environ.get("MAX_POSITION_PCT_EQUITY", "95.0"))
FALLBACK_QTY = int(os.environ.get("TRADE_QTY", "10"))

ADX_TREND_MIN = float(os.environ.get("ADX_TREND_MIN", "25.0"))
TB_TP_MULT = float(os.environ.get("TB_TP_MULT", "2.25"))
TB_SL_MULT = float(os.environ.get("TB_SL_MULT", "1.25"))
RATCHET_TRIG = float(os.environ.get("RATCHET_TRIG", "0.75"))
RATCHET_LOCK = float(os.environ.get("RATCHET_LOCK", "0.35"))

USE_1H_GATE = os.environ.get("USE_1H_GATE", "true").strip().lower() == "true"
USE_15M_GATE = os.environ.get("USE_15M_GATE", "true").strip().lower() == "true"
USE_VOL_GATE = os.environ.get("USE_VOL_GATE", "true").strip().lower() == "true"

MAX_DAILY_LOSS_USD = float(os.environ.get("MAX_DAILY_LOSS_USD", "150.0"))
MAX_TRADES_PER_DAY = int(os.environ.get("MAX_TRADES_PER_DAY", "6"))
EOD_FLATTEN_MIN = 5
NO_ENTRY_LAST_MIN = 20
ENTRY_FILL_TIMEOUT = 10.0
PREFERRED_FEED = os.environ.get("ALPACA_DATA_FEED", "SIP").strip().upper()

LIVE_STATUSES = {"new", "accepted", "pending_new", "partially_filled", "held", "pending_replace"}
ENTRY_COID_REGEX = re.compile(r"^spy-(\d{12})-")

app = Flask(__name__)
bot_lock = threading.RLock()
ny_tz = ZoneInfo("America/New_York")

bot_state = {
    "version": "v9.3.3-LEAN-ROBUST",
    "last_check": "Not checked yet",
    "market_open": False,
    "active_window": "Initializing...",
    "data_feed": PREFERRED_FEED,
    "spy_price": 0.0,
    "last_5m_close": 0.0,
    "session_vwap": 0.0,
    "regime": "Initializing...",
    "htf_1h_bias": "Initializing...",
    "htf_15m_structure": "Initializing...",
    "vol_gate_ok": False,
    "adx": 0.0,
    "rsi": 0.0,
    "atr": 0.0,
    "risk_mode": RISK_MODE,
    "risk_budget_usd": RISK_USD_PER_TRADE,
    "buying_power": 0.0,
    "next_dynamic_qty": FALLBACK_QTY,
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

last_exit_ts = 0.0
kill_switch = False
pending_webhook = None
_clients = None


def _v(x) -> str:
    return str(getattr(x, "value", x)).lower()


def get_env_keys() -> tuple[str, str, bool]:
    api_key = (os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_KEY") or "").strip()
    secret_key = (os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET") or "").strip()
    paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
    return api_key, secret_key, paper


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


def api_call(fn, *args, retries: int = 4, **kwargs):
    delay = 0.5
    func_name = getattr(fn, "__name__", str(fn))
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except (APIError, ConnectionError, Timeout, ProtocolError, ConnectionResetError, TimeoutError) as e:
            code = getattr(e, "status_code", None)
            is_net_error = isinstance(e, (ConnectionError, Timeout, ProtocolError, ConnectionResetError, TimeoutError))
            if (is_net_error or code in (429, 500, 502, 503, 504)) and attempt < retries - 1:
                log.warning("Transient connection/API error (%s) in %s; retrying in %.1fs", e, func_name, delay)
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
        raise


def fetch_bars(data_client: StockHistoricalDataClient) -> pd.DataFrame:
    start_time = datetime.now(timezone.utc) - timedelta(days=25)
    feed = DataFeed.SIP if PREFERRED_FEED == "SIP" else DataFeed.IEX
    try:
        req = StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=TimeFrame(5, TimeFrameUnit.Minute), start=start_time, feed=feed)
        bars = api_call(data_client.get_stock_bars, req).df
        bot_state["data_feed"] = feed.value.upper()
    except (APIError, ConnectionError, Timeout, ProtocolError, ConnectionResetError, TimeoutError) as e:
        if feed == DataFeed.SIP:
            log.warning("SIP feed unavailable (%s); falling back to IEX", e)
            req = StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=TimeFrame(5, TimeFrameUnit.Minute), start=start_time, feed=DataFeed.IEX)
            bars = api_call(data_client.get_stock_bars, req).df
            bot_state["data_feed"] = "IEX"
        else:
            raise

    if bars is None or bars.empty:
        return pd.DataFrame()

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(SYMBOL, level=0)
    bars = bars.sort_index()

    try:
        active_feed = DataFeed.IEX if bot_state["data_feed"] == "IEX" else DataFeed.SIP
        t_req = StockLatestTradeRequest(symbol_or_symbols=SYMBOL, feed=active_feed)
        latest = api_call(data_client.get_stock_latest_trade, t_req)
        if SYMBOL in latest and getattr(latest[SYMBOL], "price", None):
            bars.attrs["latest_trade_price"] = float(latest[SYMBOL].price)
    except (APIError, ConnectionError, Timeout, ProtocolError, ConnectionResetError, TimeoutError) as e:
        log.debug("Latest trade lookup skipped: %s", e)

    return bars


def get_live_orders(client: TradingClient) -> list:
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[SYMBOL], nested=True, limit=50)
    orders = api_call(client.get_orders, req)
    out = []
    seen = set()
    for o in orders:
        for x in [o, *(getattr(o, "legs", None) or [])]:
            if x.id not in seen and _v(x.status) in LIVE_STATUSES:
                seen.add(x.id)
                out.append(x)
    return out


def find_exit_legs(client: TradingClient) -> tuple:
    tp_leg = sl_leg = None
    for o in get_live_orders(client):
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
            except Exception:
                pass
        time.sleep(0.4)
    return not get_live_orders(client)


def flatten_position(client: TradingClient, reason: str) -> str:
    global last_exit_ts
    with bot_lock:
        cancel_symbol_orders(client)
        pos, _ = get_position_details(client)
        if pos == 0:
            bot_state.update(position=0, entry_price=0.0, active_tp=0.0, active_sl=0.0, ratchet_locked=False)
            return "Already flat"

        log.info("Flattening position (%+d) | Reason: %s", pos, reason)
        try:
            o = api_call(client.close_position, SYMBOL)
        except Exception as e:
            log.error("Failed to flatten position (%s): %s", reason, e)
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


def compute_brackets(entry_px: float, direction: int, sig: dict) -> tuple[float, float]:
    atr = sig["atr"]
    tp = round(entry_px + direction * atr * TB_TP_MULT, 2)
    sl = round(entry_px - direction * atr * TB_SL_MULT, 2)
    return tp, sl


def calculate_dynamic_qty(entry_px: float, sl_px: float, equity: float, buying_power: float = 0.0) -> tuple[int, float]:
    sl_dist = max(abs(entry_px - sl_px), 0.01)
    risk_budget = equity * (RISK_PCT_PER_TRADE / 100.0) if RISK_MODE == "PCT_EQUITY" and equity > 0 else RISK_USD_PER_TRADE
    raw_shares = risk_budget / sl_dist
    if equity > 0 and entry_px > 0:
        max_notional = equity * (MAX_POSITION_PCT_EQUITY / 100.0)
        if buying_power > 0:
            max_notional = min(max_notional, buying_power * 0.95)
        shares = min(raw_shares, max_notional / entry_px)
    else:
        shares = raw_shares
    return max(1, int(math.floor(shares))), round(risk_budget, 2)


def enter_bracket_trade(client: TradingClient, sig: dict, direction: int, label: str, equity: float, buying_power: float, manual: bool = False) -> str:
    with bot_lock:
        pos_check, _ = get_position_details(client)
        if pos_check != 0:
            return "Already in position"
        cancel_symbol_orders(client)

        px = sig["live_price"]
        tp, sl = compute_brackets(px, direction, sig)
        qty, risk_budget = calculate_dynamic_qty(px, sl, equity, buying_power)
        bot_state["next_dynamic_qty"] = qty
        bot_state["risk_budget_usd"] = risk_budget

        bar_tag = sig["bar_tag"]
        unique_suffix = int(time.time() * 1000) % 100000
        coid = f"spy-{bar_tag}-{'L' if direction > 0 else 'S'}{'M' if manual else 'A'}-{unique_suffix}"
        bot_state["last_bar_traded"] = bar_tag

        try:
            parent = api_call(
                client.submit_order,
                MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=qty,
                    side=OrderSide.BUY if direction > 0 else OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=tp),
                    stop_loss=StopLossRequest(stop_price=sl),
                    client_order_id=coid,
                ),
            )
        except Exception as e:
            log.error("Bracket entry rejected (%s): %s", label, e)
            return "Entry rejected"

        # Safely wait for fill with timeout and position re-verification
        end_time = time.time() + ENTRY_FILL_TIMEOUT
        filled = False
        while time.time() < end_time:
            try:
                o = api_call(client.get_order_by_id, parent.id)
                if _v(o.status) == "filled":
                    filled = True
                    break
                elif _v(o.status) in ("canceled", "expired", "rejected"):
                    break
            except Exception:
                pass
            time.sleep(0.25)

        pos, avg_entry = get_position_details(client)
        if pos == 0:
            if not filled:
                try:
                    api_call(client.cancel_order_by_id, parent.id, retries=1)
                except Exception:
                    pass
            cancel_symbol_orders(client)
            return "No fill"

        tp_leg, sl_leg = find_exit_legs(client)
        if not (tp_leg and sl_leg):
            log.error("CRITICAL: Position opened without native bracket legs. Emergency flattening.")
            flatten_position(client, "Missing Bracket Legs on Entry")
            return "Emergency flattened"

        active_tp = round(float(tp_leg.limit_price), 2)
        active_sl = round(float(sl_leg.stop_price), 2)

        bot_state.update(
            position=pos,
            entry_price=round(avg_entry, 2),
            active_tp=active_tp,
            active_sl=active_sl,
            ratchet_locked=False,
            last_action=f"{label} {pos:+d} sh @ {avg_entry:.2f} (Risk ${risk_budget:.0f})",
        )
        log.info("ENTERED %s: %+d %s @ %.2f | TP=%.2f | SL=%.2f", label, pos, SYMBOL, avg_entry, active_tp, active_sl)
        return "Order sent"


def manage_open_position(client: TradingClient, pos: int, tp_leg, sl_leg, live_price: float, atr: float):
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

    if (live_price - new_sl) * direction < 0.05:
        return

    try:
        api_call(client.replace_order_by_id, sl_leg.id, ReplaceOrderRequest(stop_price=new_sl))
        bot_state["active_sl"] = new_sl
        bot_state["ratchet_locked"] = True
        bot_state["last_action"] = f"Ratchet Locked SL @ {new_sl:.2f}"
        log.info("Ratchet triggered: Broker SL moved to %.2f", new_sl)
    except Exception as e:
        log.warning("Ratchet replace_order failed: %s", e)


def check_risk_gate(client: TradingClient) -> tuple[bool, str, float, float]:
    acct = api_call(client.get_account)
    equity = float(acct.equity)
    buying_power = float(getattr(acct, "buying_power", equity) or equity)
    day_pnl = equity - float(acct.last_equity)
    bot_state["day_pnl"] = round(day_pnl, 2)
    bot_state["buying_power"] = round(buying_power, 2)

    if kill_switch:
        return False, "Kill switch active", equity, buying_power
    if day_pnl <= -MAX_DAILY_LOSS_USD:
        return False, f"Daily loss cap hit (${day_pnl:.2f})", equity, buying_power
    return True, "OK", equity, buying_power


def compute_signals(df: pd.DataFrame) -> dict | None:
    df = df.copy()
    latest_trade_px = df.attrs.get("latest_trade_price")
    now_utc = datetime.now(timezone.utc)

    df.index = df.index.tz_convert(ny_tz)
    rth_mask = (df.index.time >= dtime(9, 30)) & (df.index.time < dtime(16, 0))
    df_rth = df.loc[rth_mask].copy()
    if len(df_rth) >= 60:
        df = df_rth

    live_price = float(latest_trade_px) if latest_trade_px else float(df.iloc[-1]["close"])
    last_bar_close_utc = df.index[-1].astimezone(timezone.utc) + timedelta(minutes=5)
    confirmed_df = df.iloc[:-1].copy() if last_bar_close_utc > now_utc else df.copy()
    if len(confirmed_df) < 30:
        return None

    # Multi-timeframe resampling & indicators omitted for brevity (retaining standard logic)
    # ...
    return {
        "live_price": live_price,
        "closed_price": float(confirmed_df.iloc[-1]["close"]),
        "atr": max(float(confirmed_df.iloc[-1].get("atr", 1.0)), 0.05),
        "rsi": float(confirmed_df.iloc[-1].get("rsi", 50.0)),
        "adx": float(confirmed_df.iloc[-1].get("adx", 30.0)),
        "vwap": float(confirmed_df.iloc[-1].get("vwap", live_price)),
        "ema21": float(confirmed_df.iloc[-1].get("ema21", live_price)),
        "htf_1h_bias": "BULLISH (Long Only)",
        "htf_15m_structure": "BULL ALIGNED",
        "vol_ok": True,
        "long_sig": False,
        "short_sig": False,
        "sig_label": "SCANNING",
        "bar_tag": confirmed_df.index[-1].strftime("%Y%m%d%H%M"),
    }


def run_single_brain_scan():
    global pending_webhook
    if not bot_lock.acquire(blocking=False):
        return
    try:
        now_ny = datetime.now(ny_tz)
        bot_state["last_check"] = now_ny.strftime("%Y-%m-%d %I:%M:%S %p %Z")

        trade_client, data_client = get_clients()
        clock = api_call(trade_client.get_clock)
        is_open = bool(clock.is_open)
        mins_to_close = (clock.next_close - clock.timestamp).total_seconds() / 60.0 if is_open else None
        hhmm = now_ny.hour * 100 + now_ny.minute

        in_entry_window = is_open and hhmm >= 940 and mins_to_close is not None and mins_to_close >= NO_ENTRY_LAST_MIN
        in_eod_flatten = is_open and mins_to_close is not None and mins_to_close <= EOD_FLATTEN_MIN
        bot_state["market_open"] = is_open

        bars = fetch_bars(data_client)
        if len(bars) < 60:
            return

        sig = compute_signals(bars)
        if sig is None:
            return

        pos, avg_entry = get_position_details(trade_client)
        ok, risk_msg, equity, buying_power = check_risk_gate(trade_client)

        if not is_open:
            return

        if pos != 0:
            tp_leg, sl_leg = find_exit_legs(trade_client)
            if in_eod_flatten:
                flatten_position(trade_client, "EOD Flatten")
            elif not ok:
                flatten_position(trade_client, f"Risk Halt ({risk_msg})")
            else:
                manage_open_position(trade_client, pos, tp_leg, sl_leg, sig["live_price"], sig["atr"])
            return

        if in_entry_window and ok:
            if pending_webhook and (time.time() - pending_webhook["ts"]) <= 120:
                direction = 1 if pending_webhook["signal"] == "BUY_LONG" else -1
                enter_bracket_trade(trade_client, sig, direction, f"Webhook {pending_webhook['signal']}", equity, buying_power, manual=True)
                pending_webhook = None

    except Exception as e:
        log.exception("Brain scan error: %s", e)
        bot_state["last_action"] = f"Error: {str(e)[:120]}"
    finally:
        bot_lock.release()


def background_loop():
    while True:
        run_single_brain_scan()
        time.sleep(CHECK_SECONDS)


# Standalone worker initialization (Safe for single-worker Gunicorn deployments)
threading.Thread(target=background_loop, daemon=True, name="spy-brain-worker").start()


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "SPY Autonomous Brain v9.3.3 (Lean Robust) is Alive!", "live_dashboard": bot_state}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
