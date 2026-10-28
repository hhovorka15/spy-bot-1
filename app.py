import hmac
import logging
import os
import time
import threading

from flask import Flask, jsonify, request
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, PositionSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spy-bot")

# Safe environment loading: won't crash Render on boot if a variable is missing
API_KEY = os.environ.get("ALPACA_KEY", "").strip()
SECRET_KEY = os.environ.get("ALPACA_SECRET", "").strip()
WEBHOOK_PASS = os.environ.get("WEBHOOK_PASSPHRASE", "MY_SECRET_PIN_2026").strip()
PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

SYMBOL = "SPY"
QTY = int(os.environ.get("TRADE_QTY", "10"))

TARGETS = {
    "BUY_LONG": QTY,
    "BUY_SHORT": -QTY,
    "EXIT_POSITION": 0,
}

app = Flask(__name__)
order_lock = threading.Lock()


def get_client():
    if not API_KEY or not SECRET_KEY:
        raise RuntimeError("ALPACA_KEY or ALPACA_SECRET is missing in Render Environment Variables!")
    return TradingClient(API_KEY, SECRET_KEY, paper=PAPER)


def current_position_qty(client):
    """Signed share count: positive = long, negative = short, 0 = flat."""
    try:
        pos = client.get_open_position(SYMBOL)
        qty = abs(int(float(pos.qty)))
        return -qty if pos.side == PositionSide.SHORT else qty
    except Exception as e:
        # 404 means no position is currently open (flat)
        if getattr(e, "status_code", None) == 404 or "position does not exist" in str(e).lower():
            return 0
        raise


@app.route("/", methods=["GET"])
def home():
    keys_ready = bool(API_KEY and SECRET_KEY)
    return f"SPY Loop Bot is Alive and Listening! (Keys configured: {keys_ready})", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True, silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Body must be a JSON object"}), 400

    supplied = str(data.get("passphrase", ""))
    if not hmac.compare_digest(supplied.encode(), WEBHOOK_PASS.encode()):
        log.warning("Rejected request: bad passphrase")
        return jsonify({"error": "Unauthorized"}), 401

    signal = data.get("signal")
    if signal not in TARGETS:
        log.warning("Rejected unknown signal: %r", signal)
        return jsonify({"error": f"Unknown signal: {signal!r}"}), 400

    target = TARGETS[signal]

    try:
        with order_lock:
            client = get_client()
            current = current_position_qty(client)
            log.info("signal=%s current=%s target=%s", signal, current, target)

            # 1. Claude's Upgrade: Ignore duplicate signals if already in target position
            if current == target:
                return jsonify({"status": "Already at target position, no order sent"}), 200

            # 2. If exiting OR flipping sides (Long -> Short or Short -> Long), close existing first
            if current != 0 and (target == 0 or (current > 0 and target < 0) or (current < 0 and target > 0)):
                log.info("Closing existing position (%s shares) before proceeding...", current)
                client.close_position(SYMBOL)
                if target == 0:
                    return jsonify({"status": "Position flattened", "closed_qty": current}), 200
                time.sleep(0.6)  # Brief pause so Alpaca settles the close before opening opposite side
                current = 0

            # 3. Submit order for the remaining target shares
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
            log.info("Submitted order id=%s %s %s %s", order.id, side.value, abs(delta), SYMBOL)
            return jsonify({"status": "Order submitted", "side": side.value, "qty": abs(delta)}), 200

    except Exception as e:
        log.exception("Order execution failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
