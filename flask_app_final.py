from flask import Flask, request, jsonify, render_template
import json
import pandas as pd
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf
from itertools import product
import yfinance as yf
import pandas as pd
import json
import os


app = Flask(__name__)


from main_function_module import main_function
from main_function_module import short_best_decision
USER_FILE = 'user_data.json'

def read_user_data():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as file:
            return json.load(file)
    return {"tickers": []}

def write_user_data(data):
    with open(USER_FILE, 'w') as file:
        json.dump(data, file)

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
        user_data['tickers'].append(ticker)
        write_user_data(user_data)
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = read_user_data()
        tickers = user_data.get('tickers', [])

        days_to_predict = int(request.form['days_to_predict'])
        investment_budget = float(request.form['investment_budget'])
        biggest_allowable_net_loss = float(request.form['biggest_allowable_net_loss'])
        allowable_stock_risk = float(request.form['allowable_stock_risk'])

        data = {
            'tickers': tickers,
            'days_to_predict': days_to_predict,
            'investment_budget': investment_budget,
            'biggest_allowable_net_loss': biggest_allowable_net_loss,
            'allowable_stock_risk': allowable_stock_risk
        }

        with open('user_data.json', 'w') as f:
            json.dump(data, f)

        best_decision, regret, i, selected_tickers, regrets, results, l = main_function(tickers, days_to_predict, investment_budget, biggest_allowable_net_loss,
                               allowable_stock_risk)
        result = short_best_decision(best_decision, selected_tickers), best_decision, regret, i, selected_tickers, regrets, results, l



        return jsonify(result)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except SyntaxError:
        return jsonify({
                           "error": "Invalid ticker format. Please enter a list of tickers in the format: ['AAPL', 'MSFT', ...]"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == '__main__':
    app.run(debug=True)
