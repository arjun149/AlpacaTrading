import numpy as np


class SMAStrategy:
    def __init__(self, window: int, capital: float):
        self.window = window
        self.capital = capital
        self.bitcoin = 0

    def evaluate(self, price, sma):
        """Decide whether to buy, sell, or hold based on SMA crossover."""
        if self.capital <= 0 and self.bitcoin <= 0:
            return "EXIT"
        if sma is None or price is None:
            return "HOLD"
        if sma > price and self.capital >= price:
            self.capital -= price
            self.bitcoin += 1
            return "BUY"
        elif sma < price and self.bitcoin >= 1:
            self.capital += price
            self.bitcoin -= 1
            return "SELL"
        return "HOLD"

    def total_value(self, price):
        """Compute total account value."""
        return self.capital + self.bitcoin * price

    def reset_position(self, price):
        """Convert all holdings to cash."""
        self.capital = self.total_value(price)
        self.bitcoin = 0

    def calculate_beta(self, strategy_returns, market_returns):
        return np.cov(strategy_returns, market_returns)[0][1] / np.var(market_returns)

    def calculate_alpha(self, strategy_returns, market_returns, beta, risk_free_rate=0.0):
        avg_strategy = np.mean(strategy_returns)
        avg_market = np.mean(market_returns)
        return avg_strategy - (risk_free_rate + beta * (avg_market - risk_free_rate))
