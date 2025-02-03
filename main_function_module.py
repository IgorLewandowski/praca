import yfinance as yf
import pandas as pd
import json



def load_tickers_and_dates(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        tickers = [stock['symbol'] for stock in data['stocks']]
        start_date = data['start_date']
        end_date = data['end_date']
    return tickers, start_date, end_date



def fetch_and_prepare_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    if stock_data.isnull().values.any():
        stock_data = stock_data.fillna(method='ffill')
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    stock_data['Daily Return'] = stock_data['Adj Close'].pct_change()
    stock_data['Ticker'] = symbol
    return stock_data.dropna()



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import yfinance as yf
from itertools import product



def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)





def predict_stock_price(ticker, days_to_predict):
    model = tf.keras.models.load_model('model.h5')

    stock_data = yf.download(ticker, period="1y")
    data = stock_data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    time_step = 100
    X, y = create_dataset(data_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    predictions = []
    last_sequence = X[-1]

    for _ in range(days_to_predict):
        prediction = model.predict(last_sequence.reshape(1, time_step, 1))
        predictions.append(prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], prediction).reshape(time_step, 1)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions_original_scale = scaler.inverse_transform(predictions)

    return predictions_original_scale


def predict_stock_price_bullish(ticker, days_to_predict, randomness=0.01):
    model = tf.keras.models.load_model('model.h5')
    stock_data = yf.download(ticker, period="1y")
    data = stock_data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    time_step = 100
    X, y = create_dataset(data_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    predictions = []
    last_sequence = X[-1]

    for _ in range(days_to_predict):
        prediction = model.predict(last_sequence.reshape(1, time_step, 1))
        prediction = prediction + (np.random.rand() * randomness)
        predictions.append(prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], prediction).reshape(time_step, 1)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions_original_scale = scaler.inverse_transform(predictions)

    return predictions_original_scale


def predict_stock_price_bearish(ticker, days_to_predict, randomness=0.01):
    model = tf.keras.models.load_model('model.h5')
    stock_data = yf.download(ticker, period="1y")
    data = stock_data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    time_step = 100
    X, y = create_dataset(data_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    predictions = []
    last_sequence = X[-1]

    for _ in range(days_to_predict):
        prediction = model.predict(last_sequence.reshape(1, time_step, 1))
        prediction = prediction - (np.random.rand() * randomness)
        predictions.append(prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], prediction).reshape(time_step, 1)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions_original_scale = scaler.inverse_transform(predictions)

    return predictions_original_scale


def generate_predictions(tickers, days_to_predict, randomness=0.01):
    predictions = {}
    for ticker in tickers:
        bullish_price = predict_stock_price_bullish(ticker, days_to_predict, randomness)[-1, 0]
        bearish_price = predict_stock_price_bearish(ticker, days_to_predict, randomness)[-1, 0]
        current_price = yf.Ticker(ticker).history(period='1d')['Close'][0]
        predicted_price = predict_stock_price(ticker, days_to_predict)[-1, 0]
        predictions[ticker] = [bullish_price, bearish_price, current_price, predicted_price]
    return predictions


def calculate_percent_change(predictions):
    percent_changes = {}
    for ticker, prices in predictions.items():
        bullish_price, bearish_price, current_price, predicted_price = prices
        bullish_change = ((bullish_price - current_price) / current_price) * 100
        bearish_change = ((bearish_price - current_price) / current_price) * 100
        predicted_change = ((predicted_price - current_price) / current_price) * 100
        percent_changes[ticker] = [bullish_change, bearish_change, predicted_change]
    return percent_changes


def stock_selection(percent_changes, allowable_change):
    selected_stocks = {}
    for ticker, changes in percent_changes.items():
        bullish_change, bearish_change, predicted_change = changes
        if bullish_change >= allowable_change and bearish_change >= allowable_change and predicted_change >= allowable_change:
            selected_stocks[ticker] = changes
    return selected_stocks


# heurystyki wybierania portfolio:
def greedy(scenario, tickers, current_prices, investment_budget, biggest_allowable_net_loss):
    num_shares = [0] * len(tickers)
    remaining_budget = investment_budget
    remaining_allowable_loss = biggest_allowable_net_loss
    total_invested = 0
    total_loss = 0

    potential_gains = {tickers[i]: scenario[i] - current_prices[i] for i in range(len(tickers))}

    sorted_stocks = sorted(potential_gains.items(), key=lambda item: item[1], reverse=True)

    for ticker, gain in sorted_stocks:
        index = tickers.index(ticker)
        current_price = current_prices[index]
        predicted_price = scenario[index]

        max_shares_by_budget = remaining_budget // current_price
        max_shares_by_loss = remaining_allowable_loss // max(1, current_price - predicted_price)
        max_shares = min(max_shares_by_budget, max_shares_by_loss)

        if max_shares > 0:
            num_shares[index] = max_shares
            invested_amount = max_shares * current_price
            total_invested += invested_amount
            if predicted_price < current_price:
                total_loss += max_shares * (current_price - predicted_price)
            remaining_budget -= invested_amount
            remaining_allowable_loss -= invested_amount

    return num_shares, total_invested, total_loss


import random
import math


def simulated_annealing(scenario, tickers, current_prices, investment_budget, biggest_allowable_net_loss,
                        initial_temperature, cooling_rate, num_iterations):
    num_shares = [0] * len(tickers)
    best_shares = list(num_shares)
    remaining_budget = investment_budget
    remaining_allowable_loss = biggest_allowable_net_loss
    total_invested = 0
    total_loss = 0

    def calculate_investment_and_loss(num_shares):
        total_invested = 0
        total_loss = 0
        for i, shares in enumerate(num_shares):
            current_price = current_prices[i]
            predicted_price = scenario[i]
            total_invested += shares * current_price
            if predicted_price < current_price:
                total_loss += shares * (current_price - predicted_price)
        return total_invested, total_loss

    for i in range(len(tickers)):
        if remaining_budget >= current_prices[i]:
            max_shares_by_budget = remaining_budget // current_prices[i]
            max_shares_by_loss = remaining_allowable_loss // max(1, current_prices[i] - scenario[i])
            max_shares = min(max_shares_by_budget, max_shares_by_loss)
            num_shares[i] = random.randint(0, max_shares)
            invested_amount = num_shares[i] * current_prices[i]
            total_invested += invested_amount
            remaining_budget -= invested_amount
            if scenario[i] < current_prices[i]:
                total_loss += num_shares[i] * (current_prices[i] - scenario[i])
                remaining_allowable_loss -= num_shares[i] * (current_prices[i] - scenario[i])

    best_shares = list(num_shares)
    best_invested, best_loss = calculate_investment_and_loss(best_shares)

    temperature = initial_temperature

    for iteration in range(num_iterations):
        new_shares = list(num_shares)
        i = random.randint(0, len(tickers) - 1)
        if new_shares[i] > 0:
            new_shares[i] -= 1
        else:
            new_shares[i] += 1

        new_invested, new_loss = calculate_investment_and_loss(new_shares)
        if new_invested <= investment_budget and new_loss <= biggest_allowable_net_loss:
            delta = new_loss - total_loss
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                num_shares = new_shares
                total_invested, total_loss = new_invested, new_loss

                if total_loss < best_loss:
                    best_shares = list(num_shares)
                    best_invested, best_loss = total_invested, total_loss

        temperature *= cooling_rate

    return best_shares, best_invested, best_loss


import numpy as np


def ant_colony_optimization(scenario, tickers, current_prices, investment_budget, biggest_allowable_net_loss):
    num_ants = 10
    num_iterations = 100
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.5
    Q = 100

    num_stocks = len(tickers)
    pheromone = np.ones(num_stocks)
    best_solution = None
    best_invested = 0
    best_loss = float('inf')

    def fitness(shares):
        total_invested = sum(shares[i] * current_prices[i] for i in range(num_stocks))
        total_loss = sum(shares[i] * max(0, current_prices[i] - scenario[i]) for i in range(num_stocks))
        return total_invested, total_loss

    for _ in range(num_iterations):
        solutions = []
        investments = []
        losses = []

        for _ in range(num_ants):
            remaining_budget = investment_budget
            remaining_allowable_loss = biggest_allowable_net_loss
            shares = [0] * num_stocks
            for _ in range(num_stocks):
                probabilities = [
                    (pheromone[i] ** alpha) * ((scenario[i] - current_prices[i]) ** beta)
                    if remaining_budget >= current_prices[i]
                    else 0
                    for i in range(num_stocks)
                ]
                probabilities_sum = sum(probabilities)
                if probabilities_sum == 0:
                    break
                probabilities = [p / probabilities_sum for p in probabilities]
                chosen_stock = np.random.choice(num_stocks, p=probabilities)
                max_shares_by_budget = remaining_budget // current_prices[chosen_stock]
                max_shares_by_loss = remaining_allowable_loss // max(1, current_prices[chosen_stock] - scenario[chosen_stock])
                max_shares = min(max_shares_by_budget, max_shares_by_loss)
                if max_shares > 0:
                    shares[chosen_stock] += max_shares
                    remaining_budget -= max_shares * current_prices[chosen_stock]
                    remaining_allowable_loss -= max_shares * (current_prices[chosen_stock] - scenario[chosen_stock])

            total_invested, total_loss = fitness(shares)
            solutions.append(shares)
            investments.append(total_invested)
            losses.append(total_loss)

            if total_loss < best_loss and total_invested <= investment_budget:
                best_solution = shares
                best_invested = total_invested
                best_loss = total_loss

        pheromone *= (1 - evaporation_rate)
        for i in range(num_ants):
            for j in range(num_stocks):
                pheromone[j] += Q / (1 + losses[i]) if solutions[i][j] > 0 else 0

    return best_solution, best_invested, best_loss


def scenario_for_portfolio(portfolio_decision, scenario):
    resulting_values = [portfolio_decision[i] * scenario[i] for i in range(len(portfolio_decision))]
    return resulting_values


def scenarios_for_portfolios(portfolio_decisions, scenarios):
    all_results = []
    for portfolio_decision in portfolio_decisions:
        for scenario in scenarios:
            result = scenario_for_portfolio(portfolio_decision, scenario)
            all_results.append(result)

    return all_results


def result_decision(all_results):
    results = []
    for result in all_results:
        sum = 0
        for value in result:
            sum += value
        results.append(sum)
    return results


def maximum_regret_decision(decisions, results, investment_budget):
    #najlepszy wynik dla najlepszej decyzji, nie znamy wiec zakładamy bardzo wysoki
    best_val_best_dec = 1000*investment_budget
    regrets=[]
    r = 0
    for d in range(len(decisions)):
        max_regret = 0
        for i in range(3):
            regret = best_val_best_dec - results[r]
            if regret >= max_regret:
                max_regret = regret
            i+=1
            r+=1
        regrets.append(max_regret)
    return regrets

def best_decision_minmax_regret(decisions, regrets):
    min_max_regret = math.inf
    best_decision = 0
    no_best_decision = 0
    for i in range(len(regrets)):
        if regrets[i] <= min_max_regret:
            min_max_regret = regrets[i]
            best_decision = decisions[i]
            no_best_decision = i
    return best_decision, min_max_regret, no_best_decision

def select_tickers(selected_stocks, tickers):
    selected_tickers = [ticker for ticker in tickers if ticker in selected_stocks.keys()]
    return selected_tickers


def short_best_decision(best_decision, selected_tickers):
    #best_decisions = [(ticker, value) for ticker, value in zip(selected_tickers, best_decision) if value != 0]
    #r = []
    #for ticker, value in best_decisions:
    #    r.append([ticker, value])

    import copy

    # Deep copy to ensure nested structures are also copied
    decisions = copy.deepcopy(list(zip(selected_tickers, best_decision)))

    # Filter non-zero entries and format them
    return [[ticker, value] for ticker, value in decisions if value != 0]
    #return r


def main_function(tickers, days_to_predict, investment_budget, biggest_allowable_net_loss, allowable_stock_risk):
    randomness = 0.015
    predicted_prices = generate_predictions(tickers, days_to_predict, randomness)
    percent_changes = calculate_percent_change(predicted_prices)
    selected_stocks = stock_selection(percent_changes, allowable_stock_risk)
    selected_tickers = select_tickers(selected_stocks, tickers)
    sc_predicted = []
    sc_bullish = []
    sc_bearish = []
    current_prices = []
    for ticker in selected_tickers:
        if ticker in predicted_prices:
            sc_bullish.append(predicted_prices[ticker][0])
            sc_bearish.append(predicted_prices[ticker][1])
            sc_predicted.append(predicted_prices[ticker][3])
            current_prices.append(predicted_prices[ticker][2])
    portfolio_aco_bearish = ant_colony_optimization(sc_bearish, selected_tickers, current_prices, investment_budget,
                                                    biggest_allowable_net_loss)
    portfolio_aco_bullish = ant_colony_optimization(sc_bullish, selected_tickers, current_prices, investment_budget,
                                                    biggest_allowable_net_loss)
    portfolio_aco_predicted = ant_colony_optimization(sc_predicted, selected_tickers, current_prices, investment_budget,
                                                      biggest_allowable_net_loss)
    portfolio_sa_bearish = simulated_annealing(sc_bearish, selected_tickers, current_prices, investment_budget,
                                               biggest_allowable_net_loss, 40, 0.01, 100)
    portfolio_sa_bullish = simulated_annealing(sc_bullish, selected_tickers, current_prices, investment_budget,
                                               biggest_allowable_net_loss, 40, 0.01, 100)
    portfolio_sa_predicted = simulated_annealing(sc_predicted, selected_tickers, current_prices, investment_budget,
                                                 biggest_allowable_net_loss, 40, 0.01, 100)
    portfolio_greedy_bearish = greedy(sc_bearish, selected_tickers, current_prices, investment_budget,
                                      biggest_allowable_net_loss)
    portfolio_greedy_bullish = greedy(sc_bullish, selected_tickers, current_prices, investment_budget,
                                      biggest_allowable_net_loss)
    portfolio_greedy_predicted = greedy(sc_predicted, selected_tickers, current_prices, investment_budget,
                                        biggest_allowable_net_loss)
    scenarios = [sc_predicted, sc_bullish, sc_bearish]
    portfolio_decisions = [portfolio_aco_bearish[0], portfolio_aco_bullish[0], portfolio_aco_predicted[0],
                           portfolio_sa_bearish[0], portfolio_sa_bullish[0], portfolio_sa_predicted[0],
                           portfolio_greedy_bearish[0], portfolio_greedy_bullish[0], portfolio_greedy_predicted[0]]
    l = scenarios_for_portfolios(portfolio_decisions, scenarios)
    results = result_decision(l)
    regrets = maximum_regret_decision(portfolio_decisions, results, investment_budget)
    best_decision, regret, i = best_decision_minmax_regret(portfolio_decisions, regrets)
    return best_decision, regret, i, selected_tickers, regrets, results, l






