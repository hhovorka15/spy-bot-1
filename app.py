import hmac
import logging
import os
import threading

from flask import Flask, jsonify, request
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, PositionSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spy-bot")


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


API_KEY = require_env("ALPACA_KEY")
SECRET_KEY = require_env("ALPACA_SECRET")
WEBHOOK_PASS = require_env("WEBHOOK_PASSPHRASE")
PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

SYMBOL = "SPY"
QTY = int(os.environ.get("TRADE_QTY", "10"))

TARGETS = {
    "BUY_LONG": QTY,
    "BUY_SHORT": -QTY,
    "EXIT_POSITION": 0,
}

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
app = Flask(__name__)
order_lock = threading.Lock()  # one signal processed at a time


def current_position_qty():
    """Signed share count: positive = long, negative = short, 0 = flat."""
    try:
        pos = trading_client.get_open_position(SYMBOL)
    except Exception as e:
        if getattr(e, "status_code", None) == 404:  # no open position
            return 0
        raise  # anything else is a real error; don't guess
    qty = abs(int(float(pos.qty)))
    return -qty if pos.side == PositionSide.SHORT else qty


@app.route("/", methods=["GET"])
def home():
    return "SPY bot is alive and listening.", 200


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
            current = current_position_qty()
            delta = target - current
            log.info("signal=%s current=%s target=%s delta=%s", signal, current, target, delta)

            if delta == 0:
                return jsonify({"status": "Already at target, no order sent"}), 200

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            order = trading_client.submit_order(
                MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=abs(delta),
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            )
            log.info("Submitted order id=%s %s %s %s", order.id, side.value, abs(delta), SYMBOL)
            return jsonify({"status": "Order submitted", "side": side.value, "qty": abs(delta)}), 200

    except Exception:
        log.exception("Order failed")
        return jsonify({"error": "Order failed, check server logs"}), 500


if __name__ == "__main__":  # local testing only; Render uses gunicorn
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
