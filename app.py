import hmac
import logging
import os
import time
import threading
from datetime import datetime, timedelta, timezone
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
# 1. ENVIRONMENT & STRATEGY SETTINGS
# ==========================================
API_KEY = os.environ.get("ALPACA_KEY", "").strip()
SECRET_KEY = os.environ.get("ALPACA_SECRET", "").strip()
WEBHOOK_PASS = os.environ.get("WEBHOOK_PASSPHRASE", "MY_SECRET_PIN_2026").strip()
PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

SYMBOL = "SPY"
QTY = int(os.environ.get("TRADE_QTY", "10"))
CHECK_SECONDS = 60 # Scans SPY 5m chart every 60 seconds

# v7.0 Quant Loop Parameters
ADX_THRESH = 22.0
MR_BAND_MULT = 1.5
MR_TP_MULT = 0.90
MR_SL_MULT = 1.75
TB_TP_MULT = 1.35
TB_SL_MULT = 1.50
RATCHET_TRIG = 0.65
RATCHET_LOCK = 0.15

app = Flask(__name__)
order_lock = threading.Lock()

# Live Bot Memory State
bot_state = {
    "last_check": "Starting up...",
    "market_open": False,
    "spy_price": 0.0,
    "regime": "Waiting for data...",
    "adx": 0.0,
    "rsi": 0.0,
    "position": 0,
    "entry_price": 0.0,
    "active_tp": 0.0,
    "active_sl": 0.0,
    "ratchet_locked": False,
    "last_action": "None yet"
}


def get_clients():
    if not API_KEY or not SECRET_KEY:
        raise RuntimeError("ALPACA_KEY or ALPACA_SECRET is missing in Render Environment Variables!")
    trade_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
    data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    return trade_client, data_client


def current_position_qty(client):
    try:
        pos = client.get_open_position(SYMBOL)
        qty = abs(int(float(pos.qty)))
        return -qty if pos.side == PositionSide.SHORT else qty
    except Exception as e:
        if getattr(e, "status_code", None) == 404 or "position does not exist" in str(e).lower():
            return 0
        raise


def execute_target_position(client, target, reason=""):
    with order_lock:
        current = current_position_qty(client)
        if current == target:
            return "Already at target"

        if current != 0 and (target == 0 or (current > 0 and target < 0) or (current < 0 and target > 0)):
            log.info("Closing position (%s) for reason: %s", current, reason)
            client.close_position(SYMBOL)
            if target == 0:
                bot_state["position"] = 0
                bot_state["active_tp"] = 0.0
                bot_state["active_sl"] = 0.0
                bot_state["ratchet_locked"] = False
                bot_state["last_action"] = f"Exited ({reason})"
                return "Flattened"
            time.sleep(0.6)
            current = 0

        delta = target - current
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        order = client.submit_order(
            MarketOrderRequest(
                symbol=SYMBOL,
                qty=abs(delta),
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        )
        bot_state["position"] = target
        bot_state["last_action"] = f"{side.value.upper()} {abs(delta)} ({reason})"
        log.info("Order submitted: %s %s %s (%s)", side.value, abs(delta), SYMBOL, reason)
        return "Order sent"


# ==========================================
# 2. V7.0 QUANT INDICATOR MATH ENGINE
# ==========================================
def compute_v7_signals(df):
    # EMAs
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # True Range & ATR(14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ADX(14)
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / df["atr"])
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / df["atr"])
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9))
    df["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # Keltner Stretch Bands
    df["upper_band"] = df["ema21"] + (df["atr"] * MR_BAND_MULT)
    df["lower_band"] = df["ema21"] - (df["atr"] * MR_BAND_MULT)

    c = df.iloc[-1]
    prev = df.iloc[-2]
    recent3 = df.iloc[-3:]

    atr_val = float(c["atr"])
    rsi_val = float(c["rsi"])
    adx_val = float(c["adx"])
    price = float(c["close"])

    # Engine 1: 2-Step Rubberband Mean Reversion
    was_oversold = (recent3["low"] <= recent3["lower_band"]).any() and (recent3["rsi"] < 42).any()
    was_overbought = (recent3["high"] >= recent3["upper_band"]).any() and (recent3["rsi"] > 58).any()

    rb_long = was_oversold and c["close"] > c["open"] and c["close"] > prev["close"] and c["rsi"] > prev["rsi"]
    rb_short = was_overbought and c["close"] < c["open"] and c["close"] < prev["close"] and c["rsi"] < prev["rsi"]

    # Engine 2: EMA Trend Bounce
    ema_slope_up = ((c["ema21"] - df.iloc[-4]["ema21"]) / (atr_val * 3.0)) >= 0.08
    ema_slope_dn = ((df.iloc[-4]["ema21"] - c["ema21"]) / (atr_val * 3.0)) >= 0.08

    touched_pb_long = c["low"] <= (c["ema21"] + atr_val * 0.15) and c["low"] >= (c["ema50"] - atr_val * 0.25)
    touched_pb_short = c["high"] >= (c["ema21"] - atr_val * 0.15) and c["high"] <= (c["ema50"] + atr_val * 0.25)

    tb_long = c["ema9"] > c["ema21"] and c["close"] > c["ema50"] and ema_slope_up and touched_pb_long and c["close"] > c["ema9"] and c["close"] > c["open"]
    tb_short = c["ema9"] < c["ema21"] and c["close"] < c["ema50"] and ema_slope_dn and touched_pb_short and c["close"] < c["ema9"] and c["close"] < c["open"]

    # Hybrid ADX Router
    is_trend = adx_val >= ADX_THRESH
    long_sig = tb_long if is_trend else rb_long
    short_sig = tb_short if is_trend else rb_short

    return {
        "price": price,
        "atr": atr_val,
        "rsi": rsi_val,
        "adx": adx_val,
        "is_trend": is_trend,
        "long_sig": bool(long_sig),
        "short_sig": bool(short_sig)
    }


# ==========================================
# 3. 24/7 AUTONOMOUS BACKGROUND LOOP
# ==========================================
def autonomous_brain_loop():
    log.info("Autonomous v7.0 Quant Brain started!")
    while True:
        try:
            if not API_KEY or not SECRET_KEY:
                bot_state["last_check"] = "Waiting for ALPACA_KEY & ALPACA_SECRET in Render..."
                time.sleep(15)
                continue

            trade_client, data_client = get_clients()
            clock = trade_client.get_clock()
            bot_state["market_open"] = bool(clock.is_open)
            bot_state["last_check"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            # Fetch latest 5-minute SPY bars (IEX free feed)
            start_time = datetime.now(timezone.utc) - timedelta(days=5)
            req = StockBarsRequest(
                symbol_or_symbols=SYMBOL,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start_time,
                feed=DataFeed.IEX
            )
            bars = data_client.get_stock_bars(req).df
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(SYMBOL, level=0)

            if len(bars) >= 55:
                sig = compute_v7_signals(bars)
                price = sig["price"]
                atr = sig["atr"]
                bot_state["spy_price"] = round(price, 2)
                bot_state["rsi"] = round(sig["rsi"], 1)
                bot_state["adx"] = round(sig["adx"], 1)
                bot_state["regime"] = f"TREND (EMA Bounce)" if sig["is_trend"] else f"RANGE (Rubberband MR)"

                pos = current_position_qty(trade_client)
                bot_state["position"] = pos

                if clock.is_open:
                    # 1. Manage Open Position Exits & Ratchet Shield
                    if pos > 0 and bot_state["active_tp"] > 0:
                        target_dist = bot_state["active_tp"] - bot_state["entry_price"]
                        if not bot_state["ratchet_locked"] and price >= (bot_state["entry_price"] + target_dist * RATCHET_TRIG):
                            bot_state["active_sl"] = round(bot_state["entry_price"] + atr * RATCHET_LOCK, 2)
                            bot_state["ratchet_locked"] = True

                        if price >= bot_state["active_tp"]:
                            execute_target_position(trade_client, 0, "Take-Profit Hit")
                        elif price <= bot_state["active_sl"]:
                            execute_target_position(trade_client, 0, "Stop-Loss / Ratchet Hit")

                    elif pos < 0 and bot_state["active_tp"] > 0:
                        target_dist = bot_state["entry_price"] - bot_state["active_tp"]
                        if not bot_state["ratchet_locked"] and price <= (bot_state["entry_price"] - target_dist * RATCHET_TRIG):
                            bot_state["active_sl"] = round(bot_state["entry_price"] - atr * RATCHET_LOCK, 2)
                            bot_state["ratchet_locked"] = True

                        if price <= bot_state["active_tp"]:
                            execute_target_position(trade_client, 0, "Take-Profit Hit")
                        elif price >= bot_state["active_sl"]:
                            execute_target_position(trade_client, 0, "Stop-Loss / Ratchet Hit")

                    # 2. Scan for New Loop Entry When Flat
                    elif pos == 0:
                        tp_mult = TB_TP_MULT if sig["is_trend"] else MR_TP_MULT
                        sl_mult = TB_SL_MULT if sig["is_trend"] else MR_SL_MULT

                        if sig["long_sig"]:
                            bot_state["entry_price"] = price
                            bot_state["active_tp"] = round(price + atr * tp_mult, 2)
                            bot_state["active_sl"] = round(price - atr * sl_mult, 2)
                            bot_state["ratchet_locked"] = False
                            execute_target_position(trade_client, QTY, bot_state["regime"])

                        elif sig["short_sig"]:
                            bot_state["entry_price"] = price
                            bot_state["active_tp"] = round(price - atr * tp_mult, 2)
                            bot_state["active_sl"] = round(price + atr * sl_mult, 2)
                            bot_state["ratchet_locked"] = False
                            execute_target_position(trade_client, -QTY, bot_state["regime"])

        except Exception as e:
            log.exception("Brain loop error: %s", e)
            bot_state["last_action"] = f"Error: {str(e)[:80]}"

        time.sleep(CHECK_SECONDS)


# Start background brain thread automatically when Gunicorn boots
brain_thread = threading.Thread(target=autonomous_brain_loop, daemon=True)
brain_thread.start()


@app.route("/", methods=["GET"])
def home():
    keys_ready = bool(API_KEY and SECRET_KEY)
    return jsonify({
        "status": "SPY Autonomous Brain v7.0 is Alive!",
        "keys_configured": keys_ready,
        "live_dashboard": bot_state
    }), 200


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True, silent=True) or {}
    supplied = str(data.get("passphrase", ""))
    if not hmac.compare_digest(supplied.encode(), WEBHOOK_PASS.encode()):
        return jsonify({"error": "Unauthorized"}), 401

    signal = data.get("signal")
    targets = {"BUY_LONG": QTY, "BUY_SHORT": -QTY, "EXIT_POSITION": 0}
    if signal not in targets:
        return jsonify({"error": "Unknown signal"}), 400

    trade_client, _ = get_clients()
    res = execute_target_position(trade_client, targets[signal], reason=f"Webhook {signal}")
    return jsonify({"status": res}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
