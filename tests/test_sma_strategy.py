from src.sma_strategy import SMAStrategy

def test_sma_buy_condition():
    s = SMAStrategy(window=60, capital=100)
    assert s.evaluate(price=90, sma=100) == "BUY"
