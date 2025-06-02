# Multi-Agent Reinforcement Learning for Portfolio Hedging 🇻🇳📉📈

This repository contains a full replication of the paper:

> **Pham, Uyen; Luu, Quoc; Tran, Hien (2021).**  
> *Multi-agent reinforcement learning approach for hedging portfolio problem.*  
> Published in _Soft Computing_, Springer.  
> [DOI: 10.1007/s00500-021-05801-6](https://doi.org/10.1007/s00500-021-05801-6)

---

## 🎯 Project Objective

The goal is to implement and reproduce the results of a **deep reinforcement learning-based hedging strategy** using a **multi-agent system** (equities + futures) trained via the **IMPALA algorithm**.  
This project is focused on Vietnamese stock market data, simulating realistic trading conditions (transaction costs, settlement delays, mark-to-market PnL).

---

## 📁 Repository Structure

| File / Notebook                | Description |
|-------------------------------|-------------|
| `data_preparation.ipynb`   | Prepares and aligns daily log-returns and prices from Ho Chi Minh (HSX) and Hanoi (HNX) stock exchanges. |
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

- Python 3.8+
- [Ray + RLlib](https://docs.ray.io/en/latest/rllib/)
- Gym
- NumPy, Pandas, Matplotlib
- TensorFlow or PyTorch (used by RLlib backend)

### Installation

```bash
pip install -r requirements.txt


