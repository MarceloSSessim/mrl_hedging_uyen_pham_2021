import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class MultiAgentTradingEnv(MultiAgentEnv):
    """
    MultiAgentTradingEnv

    Ambiente com 11 agentes, cada um responsável por negociar um ativo (ação ou contrato futuro).
    Alinhado com o paper: Pham et al. (2021)
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, price_df, log_return_df, asset_types,
                 initial_cash=1e6, transaction_fee=0.001, future_discount=0.001):
        super().__init__()

        assert price_df.shape == log_return_df.shape, "Mismatch between price and return data"
        assert len(asset_types) == price_df.shape[1], "Asset types length must match number of assets"

        self.price_df = price_df.reset_index(drop=True)
        self.returns_df = log_return_df.reset_index(drop=True)
        self.asset_types = asset_types
        self.num_assets = len(asset_types)

        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.future_discount = future_discount

        self.agent_ids = [f"agent_{i}" for i in range(self.num_assets)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._reset_env_vars()

    def seed(self, seed=None):
        np.random.seed(seed)

    def _reset_env_vars(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = np.zeros(self.num_assets)
        self.cost_basis = np.zeros(self.num_assets)
        self.pnl = np.zeros(self.num_assets)
        self.settlement_queue = [[] for _ in range(self.num_assets)]

    def reset(self, *, seed=None, options=None):
        self.seed(seed)
        self._reset_env_vars()

        obs = self._get_obs()
        infos = {agent_id: {} for agent_id in self.agent_ids}
        return obs, infos

    def _get_obs(self):
        log_returns = self.returns_df.iloc[self.current_step].values
        prices = self.price_df.iloc[self.current_step].values
        self.pnl = (prices - self.cost_basis) * self.positions

        obs = {
            agent_id: np.array([
                np.float32(log_returns[i]),
                np.float32(self.positions[i]),
                np.float32(self.pnl[i])
            ], dtype=np.float32)
            for i, agent_id in enumerate(self.agent_ids)
        }
        for agent_id, arr in obs.items():
            assert arr.shape == (3,), f"Obs shape errado para {agent_id}: {arr.shape}"

        return obs

    def step(self, actions):
        prices = self.price_df.iloc[self.current_step].values
        returns = self.returns_df.iloc[self.current_step].values

        # Liquidar ativos na fila T+2 (para equities)
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

        # Processar ações dos agentes
        for i, agent_id in enumerate(self.agent_ids):
            action = actions.get(agent_id, 0)
            price = prices[i]

            if action == 1:  # Buy
                cost = price * (1 + self.transaction_fee)
                if self.asset_types[i] == 'equity':
                    self.cash -= cost
                    self.settlement_queue[i].append((self.current_step + 2, 1, price))
                else:  # future
                    self.cash -= cost
                    total_qty = self.positions[i] + 1
                    total_cost = self.cost_basis[i] * self.positions[i] + price
                    self.positions[i] += 1
                    if total_qty != 0:
                        self.cost_basis[i] = total_cost / total_qty
                    else:
                        self.cost_basis[i] = 0.0  # ou np.nan se quiser sinalizar erro
            elif action == 2:
                price = prices[i]
                revenue = price * (1 - self.transaction_fee)

                if self.asset_types[i] == 'equity':
                    if self.positions[i] > 0:
                        self.cash += revenue
                        self.positions[i] = max(0, self.positions[i] - 1)
                else:  # future
                    self.cash += revenue
                    self.positions[i] -= 1  # permite posição negativa
                    total_qty = self.positions[i]
                    if total_qty != 0:
                        self.cost_basis[i] = total_cost / total_qty
                    else:
                        self.cost_basis[i] = 0.0  # ou np.nan se quiser sinalizar erro


        self.current_step += 1
        episode_ended = (
            self.current_step >= len(self.price_df) - 1 or self.cash < 0
        )

        # Recompensas
        rewards = {}
        for i, agent_id in enumerate(self.agent_ids):
            reward = returns[i] * self.positions[i]
            if self.asset_types[i] == 'future':
                reward -= self.future_discount * max(self.positions[i], 0)
            rewards[agent_id] = reward

        obs = self._get_obs()
        terminateds = {agent_id: episode_ended for agent_id in self.agent_ids}
        terminateds['__all__'] = episode_ended

        truncateds = {agent_id: False for agent_id in self.agent_ids}
        truncateds['__all__'] = False

        infos = {agent_id: {} for agent_id in self.agent_ids}
        return obs, rewards, terminateds, truncateds, infos

    def render(self):
        prices = self.price_df.iloc[self.current_step].values
        portfolio_value = self.cash + np.dot(self.positions, prices)
        print(f'Step: {self.current_step} | Portfolio Value: {portfolio_value:.2f} | Cash: {self.cash:.2f}')
