import os
from flask import Flask, request, jsonify
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

app = Flask(__name__)

# Reads your private keys safely from Render's vault
API_KEY = os.environ.get("ALPACA_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET")
WEBHOOK_PASS = "MY_SECRET_PIN_2026"

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
SYMBOL = "SPY"
QTY = 10 # Number of SPY shares to buy/sell per loop

@app.route('/', methods=['GET'])
def home():
    return "SPY Loop Bot is Alive and Listening!", 200

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(force=True)
    
    # 1. Check security password
    if data.get("passphrase") != WEBHOOK_PASS:
        return jsonify({"error": "Wrong password"}), 401

    signal = data.get("signal")

    try:
        # 2. Close any existing SPY trade first so we start fresh
        try:
            trading_client.close_position(SYMBOL)
        except Exception:
            pass

        # 3. Fire the new loop order based on TradingView's signal
        if signal == "BUY_LONG":
            order = MarketOrderRequest(symbol=SYMBOL, qty=QTY, side=OrderSide.BUY, time_in_force=TimeInForce.DAY)
            trading_client.submit_order(order)
            return jsonify({"status": "Bought SPY Long"}), 200

        elif signal == "BUY_SHORT":
            order = MarketOrderRequest(symbol=SYMBOL, qty=QTY, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            trading_client.submit_order(order)
            return jsonify({"status": "Sold SPY Short"}), 200

        elif signal == "EXIT_POSITION":
            return jsonify({"status": "Exited SPY Trade"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "No action taken"}), 200

if __name__ =='__main__':
    app.run(host='0.0.0.0', port=5000)
