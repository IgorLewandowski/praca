import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time

class StockPredictionAppTests(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.get("http://127.0.0.1:5000")  # Adjust URL based on your setup
        self.wait = WebDriverWait(self.driver, 10)
        
    def tearDown(self):
        self.driver.quit()
        
    def test_add_valid_ticker(self):
        """Test adding a valid stock ticker"""
        # Find and fill the ticker input
        ticker_input = self.wait.until(
            EC.presence_of_element_located((By.ID, "new_ticker"))
        )
        ticker_input.send_keys("AAPL")
        
        # Click the add button
        add_button = self.driver.find_element(By.XPATH, "//button[text()='Add']")
        add_button.click()
        
        # Wait for the ticker to appear in the list
        time.sleep(2)  # Allow time for API call and DOM update
        ticker_list = self.driver.find_element(By.ID, "tickerList")
        ticker_items = ticker_list.find_elements(By.CLASS_NAME, "ticker-item")
        
        # Verify the ticker was added
        added_tickers = [item.text.strip().replace("×", "").strip() for item in ticker_items]
        self.assertIn("AAPL", added_tickers)
        
    def test_add_invalid_ticker(self):
        """Test adding an invalid stock ticker"""
        ticker_input = self.wait.until(
            EC.presence_of_element_located((By.ID, "new_ticker"))
        )
        ticker_input.send_keys("INVALID123")
        
        add_button = self.driver.find_element(By.XPATH, "//button[text()='Add']")
        add_button.click()
        
        # Wait for error message
        time.sleep(2)
        error_message = self.driver.find_element(By.CLASS_NAME, "error")
        self.assertTrue(error_message.is_displayed())
        

        
    def test_add_duplicate_ticker(self):
        """Test adding the same ticker twice"""
        ticker_input = self.wait.until(
            EC.presence_of_element_located((By.ID, "new_ticker"))
        )
        
        # Add ticker first time
        ticker_input.send_keys("GOOGL")
        add_button = self.driver.find_element(By.XPATH, "//button[text()='Add']")
        add_button.click()
        time.sleep(2)
        
        # Try to add same ticker again
        ticker_input.send_keys("GOOGL")
        add_button.click()
        time.sleep(2)
        
        # Count occurrences of the ticker
        ticker_list = self.driver.find_element(By.ID, "tickerList")
        ticker_items = ticker_list.find_elements(By.CLASS_NAME, "ticker-item")
        ticker_count = len([item for item in ticker_items if "GOOGL" in item.text])
        
        # Verify ticker appears only once
        self.assertEqual(ticker_count, 1)

    def test_allowable_stock_risk_input(self):
        """Test validation of Allowable Stock Risk input field"""

        risk_input = self.wait.until(
            EC.presence_of_element_located((By.ID, "allowable_stock_risk"))
        )
        predict_button = self.driver.find_element(By.ID, "predict-button")

        # Test cases for Allowable Stock Risk input
        test_cases = [
            # (input_value, should_be_valid)
            ("101", False),  # Above maximum (100)
            ("-101", False),  # Below minimum (-100)
            ("50", True),  # Valid positive number
            ("-50", True),  # Valid negative number
            ("0", True),  # Valid zero
            ("abc", False),  # Invalid non-numeric input
            ("", False),  # Empty input
            ("25.5", True),  # Valid decimal
            ("-25.5", True),  # Valid negative decimal
            ("100", True),  # Maximum value
            ("-100", True),  # Minimum value
        ]

        for value, should_be_valid in test_cases:
            # Clear previous input
            risk_input.clear()

            # Input test value
            risk_input.send_keys(value)

            # Get input validity
            is_valid = self.driver.execute_script(
                "return document.getElementById('allowable_stock_risk').validity.valid"
            )

            # Check if validation matches expected result
            self.assertEqual(
                is_valid,
                should_be_valid,
                f"Validation failed for input '{value}'. Expected valid: {should_be_valid}, Got: {is_valid}"
            )

            # If input is invalid, check if form submission is prevented
            if not should_be_valid:
                predict_button.click()
                # Verify form wasn't submitted by checking if we're still on the same page
                current_url = self.driver.current_url
                self.assertTrue(
                    current_url.endswith("/"),
                    f"Form should not submit with invalid input: {value}"
                )

            # Additional check for valid numeric range using HTML5 validation
            if value.replace(".", "").replace("-", "").isdigit():
                numeric_value = float(value)
                self.assertEqual(
                    is_valid,
                    -100 <= numeric_value <= 100,
                    f"Range validation failed for {value}"
                )

    def test_portfolio_optimization(self):
        """Test if portfolio optimization returns results"""

        # Fill in the form
        days_input = self.driver.find_element(By.ID, "days_to_predict")
        days_input.send_keys("2")

        budget_input = self.driver.find_element(By.ID, "investment_budget")
        budget_input.send_keys("10000")

        loss_input = self.driver.find_element(By.ID, "biggest_allowable_net_loss")
        loss_input.send_keys("5000")

        risk_input = self.driver.find_element(By.ID, "allowable_stock_risk")
        risk_input.send_keys("-5")

        # Submit form
        predict_button = self.driver.find_element(By.ID, "predict-button")
        predict_button.click()

        # Wait and verify results
        time.sleep(60)  # Wait for a minute to allow calculations
        stock_tiles = self.driver.find_element(By.ID, "stock-tiles")
        self.assertTrue(stock_tiles.is_displayed())


if __name__ == "__main__":
    unittest.main()
