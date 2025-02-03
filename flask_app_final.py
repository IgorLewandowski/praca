from flask import Flask, request, jsonify, render_template
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf
from itertools import product
import os
from threading import Thread
from queue import Queue
import time
from collections import OrderedDict
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DURATION = 300  # 5 minutes
calculation_cache = OrderedDict()
MAX_CACHE_ITEMS = 100

# Background calculation queue
calculation_queue = Queue()
calculation_results = {}

from main_function_module import main_function, short_best_decision

USER_FILE = 'user_data.json'


def read_user_data():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as file:
            return json.load(file)
    return {"tickers": []}


def write_user_data(data):
    with open(USER_FILE, 'w') as file:
        json.dump(data, file)


def cache_key(params):
    """Generate a unique cache key from prediction parameters"""
    return json.dumps(sorted(params.items()))


def clean_old_cache_entries():
    """Remove expired cache entries"""
    current_time = time.time()
    for key in list(calculation_cache.keys()):
        if current_time - calculation_cache[key]['timestamp'] > CACHE_DURATION:
            calculation_cache.pop(key)

    # If cache is still too large, remove oldest entries
    while len(calculation_cache) > MAX_CACHE_ITEMS:
        calculation_cache.popitem(last=False)


def background_calculator():
    """Background thread for processing calculations"""
    while True:
        try:
            params = calculation_queue.get()
            if params is None:  # Shutdown signal
                break

            key = cache_key(params)

            # Skip if we already have a recent cached result
            if key in calculation_cache:
                if time.time() - calculation_cache[key]['timestamp'] < CACHE_DURATION:
                    continue

            # Perform calculation
            try:
                result = main_function(
                    params['tickers'],
                    params['days_to_predict'],
                    params['investment_budget'],
                    params['biggest_allowable_net_loss'],
                    params['allowable_stock_risk']
                )

                # Cache the result
                calculation_cache[key] = {
                    'result': result,
                    'timestamp': time.time()
                }

                # Clean cache periodically
                clean_old_cache_entries()

            except Exception as e:
                logger.error(f"Background calculation error: {str(e)}")

        except Exception as e:
            logger.error(f"Background calculator error: {str(e)}")
        finally:
            calculation_queue.task_done()


# Start background calculation thread
calculator_thread = Thread(target=background_calculator, daemon=True)
calculator_thread.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/add_ticker', methods=['POST'])
def add_ticker():
    data = request.get_json()
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({"success": False, "message": "Ticker is required"}), 400

    user_data = read_user_data()
    if ticker not in user_data['tickers']:
        # Verify ticker exists
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period='1d')
            if history.empty:
                return jsonify({"success": False, "message": "Invalid ticker symbol"}), 400

            # Additional validation - check if we can get actual stock info
            info = stock.info
            if not info or not isinstance(info, dict):
                return jsonify({"success": False, "message": "Invalid ticker symbol"}), 400

            user_data['tickers'].append(ticker)
            write_user_data(user_data)
        except Exception as e:
            logger.error(f"Error verifying ticker {ticker}: {str(e)}")
            return jsonify({"success": False, "message": "Failed to verify ticker"}), 400

    return jsonify({"success": True, "tickers": user_data['tickers']})


@app.route('/remove_ticker', methods=['POST'])
def remove_ticker():
    data = request.get_json()
    ticker = data.get('ticker')
    if not ticker:
        return jsonify({"success": False, "message": "Ticker is required"}), 400

    user_data = read_user_data()
    if ticker in user_data['tickers']:
        user_data['tickers'].remove(ticker)
        write_user_data(user_data)
    return jsonify({"success": True, "tickers": user_data['tickers']})


@app.route('/get_tickers', methods=['GET'])
def get_tickers():
    user_data = read_user_data()
    return jsonify({"success": True, "tickers": user_data['tickers']})


@app.route('/background_calculate')
def background_calculate():
    """Endpoint to trigger background calculation"""
    try:
        user_data = read_user_data()
        tickers = user_data.get('tickers', [])

        if not tickers:
            return jsonify({"success": False, "message": "No tickers added"}), 400

        # Get parameters from query string
        try:
            days_to_predict = int(request.args.get('days_to_predict', 0))
            investment_budget = float(request.args.get('investment_budget', 0))
            biggest_allowable_net_loss = float(request.args.get('biggest_allowable_net_loss', 0))
            allowable_stock_risk = float(request.args.get('allowable_stock_risk', 0))
        except (ValueError, TypeError):
            return jsonify({"success": False, "message": "Invalid parameters"}), 400

        # Create parameters dictionary
        params = {
            'tickers': tickers,
            'days_to_predict': days_to_predict,
            'investment_budget': investment_budget,
            'biggest_allowable_net_loss': biggest_allowable_net_loss,
            'allowable_stock_risk': allowable_stock_risk
        }

        # Add calculation to queue
        calculation_queue.put(params)

        return jsonify({"success": True, "message": "Background calculation started"})

    except Exception as e:
        logger.error(f"Background calculation error: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = read_user_data()
        tickers = user_data.get('tickers', [])

        if not tickers:
            return jsonify({"error": "No tickers added"}), 400

        # Get parameters from form
        days_to_predict = int(request.form['days_to_predict'])
        investment_budget = float(request.form['investment_budget'])
        biggest_allowable_net_loss = float(request.form['biggest_allowable_net_loss'])
        allowable_stock_risk = float(request.form['allowable_stock_risk'])

        # Update user_data with new parameters
        user_data = read_user_data()
        user_data.update({
            'days_to_predict': days_to_predict,
            'investment_budget': investment_budget,
            'biggest_allowable_net_loss': biggest_allowable_net_loss,
            'allowable_stock_risk': allowable_stock_risk
        })
        write_user_data(user_data)

        # Check cache first
        params = {
            'tickers': tickers,
            'days_to_predict': days_to_predict,
            'investment_budget': investment_budget,
            'biggest_allowable_net_loss': biggest_allowable_net_loss,
            'allowable_stock_risk': allowable_stock_risk
        }

        key = cache_key(params)

        if key in calculation_cache:
            cache_entry = calculation_cache[key]
            if time.time() - cache_entry['timestamp'] < CACHE_DURATION:
                result = cache_entry['result']
                best_decision, regret, i, selected_tickers, regrets, results, l = result
                return jsonify((short_best_decision(best_decision, selected_tickers),
                                best_decision, regret, i, selected_tickers, regrets, results, l))

        # If not in cache, perform calculation
        best_decision, regret, i, selected_tickers, regrets, results, l = main_function(
            tickers, days_to_predict, investment_budget,
            biggest_allowable_net_loss, allowable_stock_risk
        )

        # Cache the result
        calculation_cache[key] = {
            'result': (best_decision, regret, i, selected_tickers, regrets, results, l),
            'timestamp': time.time()
        }

        # Clean cache
        clean_old_cache_entries()

        shortened = short_best_decision(best_decision, selected_tickers)


        return jsonify((shortened,
                        best_decision, regret, i, selected_tickers, regrets, results, l))

    except ValueError as ve:
        logger.error(f"Value error in prediction: {str(ve)}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.teardown_appcontext
def cleanup(exception=None):
    """Cleanup function to shut down background thread"""
    if calculator_thread.is_alive():
        calculation_queue.put(None)
        calculator_thread.join(timeout=1)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)