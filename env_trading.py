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
        self._agent_ids = set(self.agent_ids)
        # Define espaços individuais
        self.single_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.single_act_space = spaces.Discrete(3)

        # Define espaços multiagente exigidos pelo Ray
        self.observation_spaces = {
            agent_id: self.single_obs_space for agent_id in self.agent_ids
        }
        self.action_spaces = {
            agent_id: self.single_act_space for agent_id in self.agent_ids
        }
        def observation_space_sample(self):
            return {
                agent_id: self.single_obs_space.sample()
                for agent_id in self.agent_ids
            }

        def action_space_sample(self):
            return {
                agent_id: self.single_act_space.sample()
                for agent_id in self.agent_ids
            }

        self.observation_space = self.single_obs_space
        self.action_space = self.single_act_space
        self.portfolio_total_history = []


        self._reset_env_vars()

    def seed(self, seed=None):
        np.random.seed(seed)

    def _reset_env_vars(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.positions = np.random.randint(0, 2, size=self.num_assets)  # -1, 0 ou 1
        self.cost_basis = np.zeros(self.num_assets)
        self.pnl = np.zeros(self.num_assets)
        self.settlement_queue = [[] for _ in range(self.num_assets)]
        self.portfolio_total_history = []
        self.last_actions = [None for _ in range(self.num_assets)]



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
            total_cost = 0.0
            total_qty = 0.0
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
                        self.positions[i] -= 1
                else:  # future
                    # Antes da venda
                    old_qty = self.positions[i]
                    old_total_cost = self.cost_basis[i] * old_qty if old_qty != 0 else 0.0

                    # Atualiza posição
                    self.cash += revenue
                    self.positions[i] -= 1

                    # Após a venda
                    new_qty = self.positions[i]
                    new_total_cost = old_total_cost - price

                    if new_qty != 0:
                        self.cost_basis[i] = new_total_cost / new_qty
                    else:
                        self.cost_basis[i] = 0.0


        self.current_step += 1
        prices = self.price_df.iloc[self.current_step].values
        portfolio_value_total = self.cash + np.dot(self.positions, prices)
        self.portfolio_total_history.append(portfolio_value_total)

        episode_ended = (
            self.current_step >= len(self.price_df) - 1 or self.cash < 0
        )

        # Recompensas
        # Recompensas
        rewards = {}
        prices = self.price_df.iloc[self.current_step].values  # preços do timestep atual

        for i, agent_id in enumerate(self.agent_ids):
            # Reward contínua: retorno da posição atual
            reward = returns[i] * self.positions[i]

            # Penalidade para posições longas em futuros
            if self.asset_types[i] == 'future':
                reward -= self.future_discount * max(self.positions[i], 0)

            # Penalidade leve por não operar (ajuda a sair da inércia)
            if self.positions[i] == 0:
                reward -= 0.001
            
            # Penaliza ação repetida (inércia)
            if hasattr(self, "last_actions") and action == self.last_actions[i]:
                reward -= 0.0005

            # Atualiza última ação
            self.last_actions[i] = action

            rewards[agent_id] = reward

        # Bônus/punição terminal por desempenho individual
        alpha = 0.001
        beta = 0.002
        if episode_ended:
            for i, agent_id in enumerate(self.agent_ids):
                final_value = self.positions[i] * prices[i]
                if final_value > 0:
                    rewards[agent_id] += alpha * final_value
                else:
                    rewards[agent_id] -= beta * abs(final_value)


        obs = self._get_obs()
        terminateds = {agent_id: episode_ended for agent_id in self.agent_ids}
        terminateds['__all__'] = episode_ended

        truncateds = {agent_id: False for agent_id in self.agent_ids}
        truncateds['__all__'] = False

        # Recupera os preços atuais
        prices = self.price_df.iloc[self.current_step].values

        # Para cada agente, calcula valor de portfólio estimado (cash dividido igualmente + valor da posição)
        infos = {
            agent_id: {
                "portfolio_value": self.cash / self.num_assets + self.positions[i] * prices[i]
            }
            for i, agent_id in enumerate(self.agent_ids)
        }
        return obs, rewards, terminateds, truncateds, infos

    def render(self):
        prices = self.price_df.iloc[self.current_step].values
        portfolio_value = self.cash + np.dot(self.positions, prices)
        print(f'Step: {self.current_step} | Portfolio Value: {portfolio_value:.2f} | Cash: {self.cash:.2f}')
