# Multi-Agent Reinforcement Learning for Portfolio Hedging 🇺🇸📉📈

This repository contains a full replication and adaptation of the paper:

> **Pham, Uyen; Luu, Quoc; Tran, Hien (2021).**  
> *Multi-agent reinforcement learning approach for hedging portfolio problem.*  
> Published in _Soft Computing_, Springer.  
> [DOI: 10.1007/s00500-021-05801-6](https://doi.org/10.1007/s00500-021-05801-6)

---

## 🎯 Project Objective

The goal is to replicate and evaluate a **deep reinforcement learning-based hedging strategy** using a **multi-agent system** trained via the **IMPALA algorithm**, as proposed by Pham et al. (2021).  
This implementation adapts the original study to use **publicly available data from the U.S. stock market**, simulating realistic trading conditions (transaction costs, settlement delays, mark-to-market PnL).

---

## 💡 US Market Adaptation

Instead of Vietnamese stock market data, this replication uses a portfolio composed of major U.S. equities and one index-tracking ETF as the hedging instrument:

```python
TICKERS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "TSLA",  # Tesla
    "NVDA",  # NVIDIA
    "META",  # Meta Platforms (Facebook)
    "INTC",  # Intel
    "NFLX",  # Netflix
    "XOM",   # Exxon Mobil
    "SPY"    # S&P 500 ETF (used as a hedge asset)
]

```
---
## 📁 Repository Structure

| File / Notebook                | Description |
|-------------------------------|-------------|
| `data_preparation.ipynb`   |Downloads and preprocesses daily log-returns and prices for selected U.S. stocks and the SPY ETF. |
| `env_trading.py`           | Custom `gym.Env` environment implementing realistic multi-agent trading logic, aligned with Algorithm 1 from the paper. |
| `model_architecture.py`    | Defines a shared LSTM-based neural network architecture for the agents (policy and value heads). |
| `train_impala.py`          | Training script using **RLlib**'s implementation of IMPALA with distributed actors and V-trace updates. |
| `evaluate_agent.ipynb`     | Evaluates a trained agent on out-of-sample data and simulates trading behavior and return series. |
| `experiments_analysis.ipynb` | Compares the RL hedging strategy against a buy-and-hold baseline, plotting cumulative returns and risk metrics. |

---

## 🧠 Key Features

- ✅ Multi-agent setting: `equity_trader` and `future_hedger`
- ✅ Realistic constraints: T+2 (equity) and T+0 (futures) settlement logic
- ✅ Log-return based observations; raw prices used for portfolio simulation
- ✅ Transaction fees and future hedging penalty modeled
- ✅ Environment aligned with **Algorithm 1** of the paper
- ✅ Compatible with **Ray + RLlib** for large-scale training (IMPALA)

---

## 🚀 Getting Started

### Requirements

- Python 3.10
- [Ray + RLlib](https://docs.ray.io/en/latest/rllib/)
- Gym
- NumPy, Pandas, Matplotlib
- TensorFlow or PyTorch (used by RLlib backend)

### Installation

```bash
pip install -r requirements.txt


