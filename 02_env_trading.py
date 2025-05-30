import gym
import numpy as np
from gym import spaces

class MultiAgentTradingEnv(gym.Env):
    """
    Multi-Agent Trading Environment using real prices for trade simulation
    and log-returns as input for RL agent.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, price_df, log_return_df, asset_types, initial_cash=1e6, transaction_fee=0.001, future_discount=0.001):
        super(MultiAgentTradingEnv, self).__init__()

        assert price_df.shape == log_return_df.shape, "Mismatch between price and return data"

        self.price_df = price_df.reset_index(drop=True)
        self.returns_df = log_return_df.reset_index(drop=True)
        self.asset_types = asset_types
        self.num_assets = self.price_df.shape[1]

        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.future_discount = future_discount

        self.action_space = spaces.MultiDiscrete([3] * self.num_assets)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_assets * 3,), dtype=np.float32
        )

        self.settlement_queue = [[] for _ in range(self.num_assets)]
        self.history = {"portfolio_value": [], "cash": [], "step": []} 
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = np.zeros(self.num_assets)
        self.cost_basis = np.zeros(self.num_assets)
        self.pnl = np.zeros(self.num_assets)
        self.settlement_queue = [[] for _ in range(self.num_assets)]
        self.history = {"portfolio_value": [], "cash": [], "step": []} 

        self.trades = {}

        return self._get_obs()

    def _get_obs(self):
        log_returns = self.returns_df.iloc[self.current_step].values
        current_prices = self.price_df.iloc[self.current_step].values
        self.pnl = (current_prices - self.cost_basis) * self.positions 
        return np.concatenate([log_returns, self.positions, self.pnl], axis=0)

    def step(self, actions):
        prices = self.price_df.iloc[self.current_step].values
        returns = self.returns_df.iloc[self.current_step].values
        reward = 0.0

      
        for i in range(self.num_assets):
            new_queue = []
            for step_due, qty, price in self.settlement_queue[i]:
                if self.current_step >= step_due:
                    total_qty = self.positions[i] + qty
                    if total_qty > 0:
                        total_cost = self.cost_basis[i] * self.positions[i] + price * qty
                        self.cost_basis[i] = total_cost / total_qty
                    else:
                        self.cost_basis[i] = 0
                    self.positions[i] += qty
                else:
                    new_queue.append((step_due, qty, price))
            self.settlement_queue[i] = new_queue


        for i in range(self.num_assets):
            action = actions[i]
            price = prices[i]
            asset_type = self.asset_types[i]

            if action == 1:  # Buy / Long
                self.cash -= price * (1 + self.transaction_fee)
                if asset_type == 'future':  # T+0
                    total_qty = self.positions[i] + 1
                    total_cost = self.cost_basis[i] * self.positions[i] + price
                    self.positions[i] += 1
                    self.cost_basis[i] = total_cost / total_qty
                else:  # T+2
                    self.settlement_queue[i].append((self.current_step + 2, 1, price))

            elif action == 2:  # Sell / Short
                if asset_type == 'future':
                    self.cash += price * (1 - self.transaction_fee)
                    self.positions[i] -= 1  # pode ir para negativo
                elif self.positions[i] > 0:
                    self.cash += price * (1 - self.transaction_fee)
                    self.positions[i] -= 1
                    # custo médio permanece

        self.current_step += 1
        done = self.current_step >= len(self.price_df) - 1
        if self.cash < 0:
            done = True

        # 🔧 MODIFICAÇÃO: Penalidade proporcional a posições longas em futuros
        portfolio_return = (returns * self.positions).sum()
        for i, a_type in enumerate(self.asset_types):
            if a_type == 'future' and self.positions[i] > 0:
                reward -= self.future_discount * self.positions[i]

        reward += 1 if portfolio_return > 0 else -1

        # 🔧 MODIFICAÇÃO: Track do valor do portfólio
        portfolio_value = self.cash + np.dot(self.positions, prices)
        self.history["portfolio_value"].append(portfolio_value)
        self.history["cash"].append(self.cash)
        self.history["step"].append(self.current_step)
        self.trades.append({
            "step": self.current_step,
            "actions": actions,
            "prices": prices.copy(),
            "positions": self.positions.copy(),
            "cash": self.cash,
            "pnl": self.pnl.copy(),
            "portfolio_value": portfolio_value,
            "reward": reward
        })

        info = {
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "pnl": self.pnl.copy(),
            "step": self.current_step,
        }
        return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        current_prices = self.price_df.iloc[self.current_step].values
        portfolio_value = self.cash + np.dot(self.positions, current_prices)
        print(f'Step: {self.current_step} | Portfolio Value: {portfolio_value:.2f} | Cash: {self.cash:.2f}')