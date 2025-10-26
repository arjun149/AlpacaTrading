from src.data_utils import get_price, get_sma
from src.sma_strategy import SMAStrategy
from src.neural_model import AllocationNN
import torch
import numpy as np
import pandas as pd
import time


def main():
    model = AllocationNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    strategy1 = SMAStrategy(window=60, capital=50000000)
    strategy2 = SMAStrategy(window=120, capital=50000000)

    iteration = 0
    while True:
        price = get_price("BTC/USD")
        sma1 = get_sma("BTC/USD", strategy1.window)
        sma2 = get_sma("BTC/USD", strategy2.window)

        action1 = strategy1.evaluate(price, sma1)
        action2 = strategy2.evaluate(price, sma2)

        print(f"[SMA 1h] Action: {action1} | Capital: {strategy1.capital:.2f} | BTC: {strategy1.bitcoin}")
        print(f"[SMA 2h] Action: {action2} | Capital: {strategy2.capital:.2f} | BTC: {strategy2.bitcoin}\n")

        iteration += 1
        if iteration % 5 == 0:
            total_value1 = strategy1.total_value(price)
            total_value2 = strategy2.total_value(price)
            total = total_value1 + total_value2

            input_data = [total_value1, total_value2] + [0] * 52  # placeholder features
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            allocation = model(input_tensor)

            strategy1.reset_position(price)
            strategy2.reset_position(price)

            strategy1.capital = allocation[0].item() * total
            strategy2.capital = allocation[1].item() * total

            reward = torch.tensor([total_value1 / total, total_value2 / total], dtype=torch.float32)
            loss = loss_fn(allocation.squeeze(), reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("[REBALANCE] Neural network reallocated capital.\n")

        time.sleep(60)


if __name__ == "__main__":
    main()
