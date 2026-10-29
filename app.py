import hmac
import logging
import os
import time
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd

from flask import Flask, jsonify, request
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, PositionSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spy-brain")

# ==========================================
# 1. MASTER CONFIGURATION (v7.0 GOLD)
# ==========================================
SYMBOL = "SPY"
QTY = int(os.environ.get("TRADE_QTY", "10"))
CHECK_SECONDS = 30 # Checks TP/SL every 30s, entries on confirmed 5m bar close
COOLDOWN_MINUTES = 15 # 3 bars (15 mins) cooldown after an exit

# v7.0 Quantitative Parameters
ADX_THRESH = 22.0 # < 22 = Range (Rubberband MR), >= 22 = Trend (EMA Bounce)
MR_BAND_MULT = 1.5
MR_TP_MULT = 0.90
MR_SL_MULT = 1.75
TB_TP_MULT = 1.35
TB_SL_MULT = 1.50
RATCHET_TRIG = 0.65 # Locks profit at 65% of TP distance
RATCHET_LOCK = 0.15 # Locks +0.15x ATR above/below entry

app = Flask(__name__)
order_lock = threading.Lock()
brain_lock = threading.Lock()
ny_tz = ZoneInfo("America/New_York")

bot_state = {
    "last_check": "Not checked yet",
    "market_open": False,
    "active_window": "Checking...",
    "spy_price": 0.0,
    "last_5m_close": 0.0,
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
    "last_action": "System booted cleanly"
}

last_exit_ts = 0.0


def get_env_keys():
    api_key = os.environ.get("ALPACA_KEY", "").strip()
    secret_key = os.environ.get("ALPACA_SECRET", "").strip()
    paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
    return api_key, secret_key, paper


def get_clients():
    api_key, secret_key, paper = get_env_keys()
    if not api_key or not secret_key:
        raise RuntimeError("Missing ALPACA_KEY or ALPACA_SECRET in Render Environment tab!")
    return TradingClient(api_key, secret_key, paper=paper), StockHistoricalDataClient(api_key, secret_key)


def get_position_details(client):
    """Returns (signed_qty, avg_entry_price)."""
    try:
        pos = client.get_open_position(SYMBOL)
        qty = abs(int(float(pos.qty)))
        signed_qty = -qty if pos.side == PositionSide.SHORT else qty
        avg_entry = float(pos.avg_entry_price)
        return signed_qty, avg_entry
    except Exception as e:
        if getattr(e, "status_code", None) == 404 or "position does not exist" in str(e).lower():
            return 0, 0.0
        raise


def execute_target_position(client, target, reason=""):
    global last_exit_ts
    with order_lock:
        current, _ = get_position_details(client)
        if current == target:
            return "Already at target"

        # Cancel any stuck open orders first
        try:
            client.cancel_orders()
        except Exception:
            pass

        # Close existing position if exiting or flipping sides
        if current != 0 and (target == 0 or (current > 0 and target < 0) or (current < 0 and target > 0)):
            log.info("Closing position (%s) | Reason: %s", current, reason)
            client.close_position(SYMBOL)
            last_exit_ts = time.time()
            cooldown_dt = datetime.fromtimestamp(last_exit_ts + COOLDOWN_MINUTES * 60, ny_tz)
            bot_state["cooldown_until"] = cooldown_dt.strftime("%I:%M:%S %p %Z")

            if target == 0:
                bot_state["position"] = 0
                bot_state["entry_price"] = 0.0
                bot_state["active_tp"] = 0.0
                bot_state["active_sl"] = 0.0
                bot_state["ratchet_locked"] = False
                bot_state["last_action"] = f"Exited ({reason})"
                return "Flattened"

            # Wait up to 3 seconds for Alpaca to settle the flatten before reversing
            for _ in range(6):
                time.sleep(0.5)
                chk_qty, _ = get_position_details(client)
                if chk_qty == 0:
                    break
            current = 0

        delta = target - current
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=abs(delta),
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        )
        bot_state["position"] = target
        bot_state["last_action"] = f"{side.value.upper()} {abs(delta)} ({reason})"
        log.info("Submitted %s %s %s | Reason: %s", side.value.upper(), abs(delta), SYMBOL, reason)
        return "Order sent"


# ==========================================
# 2. V7.0 CONFIRMED-BAR QUANT ENGINE
# ==========================================
def compute_v7_signals(df):
    df = df.copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Wilder's ATR(14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # Wilder's RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Wilder's ADX(14)
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9))
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / df["atr"].replace(0, 1e-9))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9))
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # Keltner Stretch Bands
    df["upper_band"] = df["ema21"] + (df["atr"] * MR_BAND_MULT)
    df["lower_band"] = df["ema21"] - (df["atr"] * MR_BAND_MULT)

    # CRITICAL FIX:
    # live_bar (iloc[-1]) = current real-time price for instant TP/SL exits
    # closed_bar (iloc[-2]) = last officially closed 5m candle for non-repainting entries
    live_bar = df.iloc[-1]
    c = df.iloc[-2]
    prev = df.iloc[-3]
    recent3 = df.iloc[-4:-1]

    live_price = float(live_bar["close"])
    closed_price = float(c["close"])
    atr_val = float(c["atr"])
    rsi_val = float(c["rsi"])
    adx_val = float(c["adx"])
    bar_time_str = str(df.index[-2])

    # Candle body/wick strength filter (>= 25%)
    c_range = max(float(c["high"] - c["low"]), 0.01)
    bull_strength = (float(c["close"] - c["low"]) / c_range) >= 0.25
    bear_strength = (float(c["high"] - c["close"]) / c_range) >= 0.25

    # Engine 1: 2-Step Rubberband Mean Reversion (on closed bar)
    was_oversold = (recent3["low"] <= recent3["lower_band"]).any() and (recent3["rsi"] < 42).any()
    was_overbought = (recent3["high"] >= recent3["upper_band"]).any() and (recent3["rsi"] > 58).any()

    rb_long = was_oversold and c["close"] > c["open"] and c["close"] > prev["close"] and c["rsi"] > prev["rsi"] and bull_strength
    rb_short = was_overbought and c["close"] < c["open"] and c["close"] < prev["close"] and c["rsi"] < prev["rsi"] and bear_strength

    # Engine 2: EMA Trend Bounce (on closed bar)
    ema_slope_up = ((c["ema21"] - df.iloc[-5]["ema21"]) / (atr_val * 3.0)) >= 0.08
    ema_slope_dn = ((df.iloc[-5]["ema21"] - c["ema21"]) / (atr_val * 3.0)) >= 0.08

    touched_pb_long = c["low"] <= (c["ema21"] + atr_val * 0.15) and c["low"] >= (c["ema50"] - atr_val * 0.25)
    touched_pb_short = c["high"] >= (c["ema21"] - atr_val * 0.15) and c["high"] <= (c["ema50"] + atr_val * 0.25)

    tb_long = c["ema9"] > c["ema21"] and c["close"] > c["ema50"] and ema_slope_up and touched_pb_long and c["close"] > c["ema9"] and c["close"] > c["open"] and 45 < rsi_val < 72
    tb_short = c["ema9"] < c["ema21"] and c["close"] < c["ema50"] and ema_slope_dn and touched_pb_short and c["close"] < c["ema9"] and c["close"] < c["open"] and 28 < rsi_val < 55

    is_trend = adx_val >= ADX_THRESH
    long_sig = tb_long if is_trend else rb_long
    short_sig = tb_short if is_trend else rb_short

    return {
        "live_price": live_price,
        "closed_price": closed_price,
        "atr": atr_val,
        "rsi": rsi_val,
        "adx": adx_val,
        "is_trend": is_trend,
        "long_sig": bool(long_sig),
        "short_sig": bool(short_sig),
        "bar_id": bar_time_str
    }


# ==========================================
# 3. SELF-HEALING AUTONOMOUS SCANNER
# ==========================================
def run_single_brain_scan():
    if not brain_lock.acquire(blocking=False):
        return
    try:
        now_ny = datetime.now(ny_tz)
        bot_state["last_check"] = now_ny.strftime("%Y-%m-%d %I:%M:%S %p %Z")

        api_key, secret_key, _ = get_env_keys()
        if not api_key or not secret_key:
            bot_state["regime"] = "Waiting for ALPACA_KEY & ALPACA_SECRET in Render Environment tab"
            return

        trade_client, data_client = get_clients()
        clock = trade_client.get_clock()
        bot_state["market_open"] = bool(clock.is_open)

        # NY Session Windows: Entry (09:35 - 15:45), Force Flatten (15:55 - 16:00)
        hhmm = now_ny.hour * 100 + now_ny.minute
        in_entry_window = 935 <= hhmm <= 1545
        in_eod_flatten = 1555 <= hhmm <= 1600
        bot_state["active_window"] = "ACTIVE ENTRY WINDOW" if (clock.is_open and in_entry_window) else ("EOD FLATTEN" if (clock.is_open and in_eod_flatten) else "OUTSIDE ENTRY HOURS")

        start_time = datetime.now(timezone.utc) - timedelta(days=6)
        req = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_time,
            feed=DataFeed.IEX
        )
        bars = data_client.get_stock_bars(req).df
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(SYMBOL, level=0)

        if len(bars) >= 60:
            sig = compute_v7_signals(bars)
            live_price = sig["live_price"]
            atr = sig["atr"]

            bot_state["spy_price"] = round(live_price, 2)
            bot_state["last_5m_close"] = round(sig["closed_price"], 2)
            bot_state["rsi"] = round(sig["rsi"], 1)
            bot_state["adx"] = round(sig["adx"], 1)
            bot_state["atr"] = round(atr, 2)
            bot_state["regime"] = "TREND (EMA Bounce)" if sig["is_trend"] else "RANGE (Rubberband MR)"

            pos, avg_entry = get_position_details(trade_client)
            bot_state["position"] = pos

            # Self-Heal Brackets if Server Restarted While in an Open Trade
            if pos != 0 and bot_state["active_tp"] == 0.0 and avg_entry > 0:
                tp_mult = TB_TP_MULT if sig["is_trend"] else MR_TP_MULT
                sl_mult = TB_SL_MULT if sig["is_trend"] else MR_SL_MULT
                bot_state["entry_price"] = round(avg_entry, 2)
                if pos > 0:
                    bot_state["active_tp"] = round(avg_entry + atr * tp_mult, 2)
                    bot_state["active_sl"] = round(avg_entry - atr * sl_mult, 2)
                else:
                    bot_state["active_tp"] = round(avg_entry - atr * tp_mult, 2)
                    bot_state["active_sl"] = round(avg_entry + atr * sl_mult, 2)
                bot_state["last_action"] = "Restored TP/SL brackets from Alpaca open position"

            if clock.is_open:
                # 1. Force Flatten at 15:55 EST (Zero Overnight Risk)
                if in_eod_flatten and pos != 0:
                    execute_target_position(trade_client, 0, "15:55 EST EOD Flatten")
                    return

                # 2. Manage Active Long Position
                if pos > 0 and bot_state["active_tp"] > 0:
                    target_dist = bot_state["active_tp"] - bot_state["entry_price"]
                    if not bot_state["ratchet_locked"] and live_price >= (bot_state["entry_price"] + target_dist * RATCHET_TRIG):
                        bot_state["active_sl"] = round(bot_state["entry_price"] + atr * RATCHET_LOCK, 2)
                        bot_state["ratchet_locked"] = True
                    if live_price >= bot_state["active_tp"]:
                        execute_target_position(trade_client, 0, "Take-Profit Hit")
                    elif live_price <= bot_state["active_sl"]:
                        execute_target_position(trade_client, 0, "Ratchet Lock Hit" if bot_state["ratchet_locked"] else "Stop-Loss Hit")

                # 3. Manage Active Short Position
                elif pos < 0 and bot_state["active_tp"] > 0:
                    target_dist = bot_state["entry_price"] - bot_state["active_tp"]
                    if not bot_state["ratchet_locked"] and live_price <= (bot_state["entry_price"] - target_dist * RATCHET_TRIG):
                        bot_state["active_sl"] = round(bot_state["entry_price"] - atr * RATCHET_LOCK, 2)
                        bot_state["ratchet_locked"] = True
                    if live_price <= bot_state["active_tp"]:
                        execute_target_position(trade_client, 0, "Take-Profit Hit")
                    elif live_price >= bot_state["active_sl"]:
                        execute_target_position(trade_client, 0, "Ratchet Lock Hit" if bot_state["ratchet_locked"] else "Stop-Loss Hit")

                # 4. Scan for New Loop Entries (Only when Flat, Inside NY Window, Off Cooldown, and New 5m Bar)
                elif pos == 0 and in_entry_window:
                    in_cooldown = (time.time() - last_exit_ts) < (COOLDOWN_MINUTES * 60)
                    already_traded_bar = (bot_state["last_bar_traded"] == sig["bar_id"])

                    if not in_cooldown and not already_traded_bar:
                        tp_mult = TB_TP_MULT if sig["is_trend"] else MR_TP_MULT
                        sl_mult = TB_SL_MULT if sig["is_trend"] else MR_SL_MULT

                        if sig["long_sig"]:
                            bot_state["entry_price"] = round(live_price, 2)
                            bot_state["active_tp"] = round(live_price + atr * tp_mult, 2)
                            bot_state["active_sl"] = round(live_price - atr * sl_mult, 2)
                            bot_state["ratchet_locked"] = False
                            bot_state["last_bar_traded"] = sig["bar_id"]
                            execute_target_position(trade_client, QTY, f"LONG {bot_state['regime']}")

                        elif sig["short_sig"]:
                            bot_state["entry_price"] = round(live_price, 2)
                            bot_state["active_tp"] = round(live_price - atr * tp_mult, 2)
                            bot_state["active_sl"] = round(live_price + atr * sl_mult, 2)
                            bot_state["ratchet_locked"] = False
                            bot_state["last_bar_traded"] = sig["bar_id"]
                            execute_target_position(trade_client, -QTY, f"SHORT {bot_state['regime']}")

    except Exception as e:
        log.exception("Brain scan error: %s", e)
        bot_state["last_action"] = f"Error: {str(e)[:120]}"
    finally:
        brain_lock.release()


def background_loop():
    while True:
        run_single_brain_scan()
        time.sleep(CHECK_SECONDS)


threading.Thread(target=background_loop, daemon=True).start()


@app.route("/", methods=["GET"])
def home():
    run_single_brain_scan()
    api_key, secret_key, _ = get_env_keys()
    return jsonify({
        "status": "SPY Autonomous Brain v7.0 (Gold Master) is Alive!",
        "keys_configured": bool(api_key and secret_key),
        "live_dashboard": bot_state
    }), 200


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True, silent=True) or {}
    webhook_pass = os.environ.get("WEBHOOK_PASSPHRASE", "MY_SECRET_PIN_2026").strip()
    supplied = str(data.get("passphrase", ""))
    if not hmac.compare_digest(supplied.encode(), webhook_pass.encode()):
        return jsonify({"error": "Unauthorized"}), 401

    signal = data.get("signal")
    targets = {"BUY_LONG": QTY, "BUY_SHORT": -QTY, "EXIT_POSITION": 0}
    if signal not in targets:
        return jsonify({"error": "Unknown signal"}), 400

    trade_client, _ = get_clients()
    res = execute_target_position(trade_client, targets[signal], reason=f"Manual Webhook {signal}")
    return jsonify({"status": res}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
