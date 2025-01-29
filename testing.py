import unittest
import numpy as np
from main_function_module import (generate_predictions,
                                  calculate_percent_change,
                                  greedy,
                                  result_decision,
                                  scenarios_for_portfolios,
                                  main_function)


class TestPredictionSystem(unittest.TestCase):
    def setUp(self):
        # Test data
        self.tickers = ['AAPL', 'MSFT']
        self.days_to_predict = 5
        self.investment_budget = 10000
        self.biggest_allowable_net_loss = 1000
        self.current_prices = [100, 200]
        self.predicted_prices = {'AAPL': [105, 95, 100, 102],
                                 'MSFT': [210, 190, 200, 205]}
        self.allowable_stock_risk = 0.1  # 10% risk tolerance

    def test_generate_predictions(self):
        """Test prediction generation"""
        predictions = generate_predictions(self.tickers,
                                           self.days_to_predict)
        self.assertIsInstance(predictions, dict)
        self.assertEqual(len(predictions), len(self.tickers))
        for ticker in self.tickers:
            self.assertEqual(len(predictions[ticker]), 4)

    def test_calculate_percent_change(self):
        """Test percentage change calculations"""
        changes = calculate_percent_change(self.predicted_prices)
        self.assertIsInstance(changes, dict)
        for ticker in self.tickers:
            self.assertEqual(len(changes[ticker]), 3)
            # Check if changes are within reasonable range
            for change in changes[ticker]:
                self.assertGreater(change, -100)
                self.assertLess(change, 100)

    def test_greedy_algorithm(self):
        """Test greedy algorithm"""
        portfolio = greedy(self.predicted_prices['AAPL'],
                           self.tickers,
                           self.current_prices,
                           self.investment_budget,
                           self.biggest_allowable_net_loss)
        # Check budget
        total_cost = sum([shares * price for shares, price
                          in zip(portfolio[0], self.current_prices)])
        self.assertLessEqual(total_cost, self.investment_budget)
        # Check potential loss
        potential_loss = sum([shares * max(0, price - pred)
                              for shares, price, pred
                              in zip(portfolio[0],
                                     self.current_prices,
                                     [self.predicted_prices[t][1]
                                      for t in self.tickers])])
        self.assertLessEqual(potential_loss,
                             self.biggest_allowable_net_loss)

    def test_result_decision(self):
        """Test decision results calculation"""
        test_decisions = [[10, 5], [5, 10]]
        test_scenarios = [[102, 205], [98, 195], [100, 200]]
        all_results = scenarios_for_portfolios(test_decisions, test_scenarios)
        results = result_decision(all_results)
        self.assertEqual(len(results), len(test_decisions) * 3)


def test_main_function():
    # Test case setup
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    test_days = 5
    test_budget = 10000
    test_loss = 1000
    test_risk = -5  # Negative to ensure we get some stocks selected

    print("Starting test with parameters:")
    print(f"Tickers: {test_tickers}")
    print(f"Days to predict: {test_days}")
    print(f"Budget: ${test_budget}")
    print(f"Max loss: ${test_loss}")
    print(f"Risk threshold: {test_risk}%")

    try:
        # Run main function
        best_decision, regret, i, selected_tickers, regrets, results, scenarios = main_function(
            test_tickers,
            test_days,
            test_budget,
            test_loss,
            test_risk
        )

        # Print results
        print("\nTest Results:")
        print(f"Selected tickers: {selected_tickers}")
        print("\nBest portfolio allocation:")
        for ticker, shares in zip(selected_tickers, best_decision):
            if shares > 0:
                print(f"{ticker}: {shares} shares")

        print(f"\nExpected maximum regret: {regret:.2f}")
        print(
            f"Algorithm used: {['ACO Bearish', 'ACO Bullish', 'ACO Predicted', 'SA Bearish', 'SA Bullish', 'SA Predicted', 'Greedy Bearish', 'Greedy Bullish', 'Greedy Predicted'][i]}")

        # Basic checks
        print("\nValidation checks:")
        print(f"1. Portfolio contains stocks: {'Pass' if any(shares > 0 for shares in best_decision) else 'Fail'}")
        print(f"2. All shares are non-negative: {'Pass' if all(shares >= 0 for shares in best_decision) else 'Fail'}")
        print(
            f"3. Selected tickers match allocations: {'Pass' if len(selected_tickers) == len(best_decision) else 'Fail'}")
        print(f"4. Regret is calculated: {'Pass' if regret is not None else 'Fail'}")

        return True

    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        return False

if __name__ == '__main__':
    unittest.main()