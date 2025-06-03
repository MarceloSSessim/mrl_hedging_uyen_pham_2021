import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.algorithms.impala import ImpalaConfig

from env_trading import MultiAgentTradingEnv
from model_architecture import SharedLSTMModel
from utils import policy_mapping_fn

# === 1. Configurações ===
checkpoint_path = "results/impala_trading_experiment/IMPALA_MultiAgentTradingEnv-v0_d3720_00000_0_2025-06-03_12-31-32/checkpoint_000003"
price_path = "data/processed/raw_prices.csv"
return_path = "data/processed/returns_log.csv"
asset_types = ["equity"] * 10 + ["future"]
output_dir = "evaluation_outputs"
os.makedirs(output_dir, exist_ok=True)

# === 2. Registro do modelo e ambiente ===
ModelCatalog.register_custom_model("shared_lstm_model", SharedLSTMModel)

def create_env(env_config):
    price_df = pd.read_csv(env_config["price_path"], index_col=0).astype(np.float32).iloc[501:550]
    return_df = pd.read_csv(env_config["return_path"], index_col=0).astype(np.float32).iloc[501:550]
    common_columns = price_df.columns.intersection(return_df.columns)
    price_df = price_df[common_columns]
    return_df = return_df[common_columns]
    min_len = min(len(price_df), len(return_df))
    price_df = price_df.iloc[:min_len]
    return_df = return_df.iloc[:min_len]

    return MultiAgentTradingEnv(
        price_df=price_df,
        log_return_df=return_df,
        asset_types=env_config["asset_types"],
        initial_cash=env_config["initial_cash"],
        transaction_fee=env_config["transaction_fee"],
        future_discount=env_config["future_discount"],
    )

register_env("MultiAgentTradingEnv-v0", create_env)

# === 3. Define configuração do algoritmo ===
config = (
    ImpalaConfig()
    .environment(
        env="MultiAgentTradingEnv-v0",
        env_config={
            "price_path": price_path,
            "return_path": return_path,
            "asset_types": asset_types,
            "initial_cash": 1e6,
            "transaction_fee": 0.001,
            "future_discount": 0.001,
        }
    )
    .framework("torch")
    .rollouts(num_rollout_workers=0)
    .training(model={"custom_model": "shared_lstm_model"})
)

# === 4. Carrega o modelo a partir do checkpoint ===
algo = config.build()
algo.restore(checkpoint_path)
device = next(algo.get_policy("shared_policy").model.parameters()).device
print(f"✅ Avaliação usando dispositivo: {device}")

# === 5. Prepara os dados de avaliação e cria o ambiente ===
price_df = pd.read_csv(price_path, index_col=0).astype(np.float32).iloc[501:600]
return_df = pd.read_csv(return_path, index_col=0).astype(np.float32).iloc[501:600]
common_columns = price_df.columns.intersection(return_df.columns)
price_df = price_df[common_columns]
return_df = return_df[common_columns]

env = MultiAgentTradingEnv(
    price_df=price_df,
    log_return_df=return_df,
    asset_types=asset_types,
    initial_cash=1e6,
    transaction_fee=0.001,
    future_discount=0.001,
)

# === 6. Executa um episódio e armazena recompensas e portfólio ===
obs, _ = env.reset()
done = {"__all__": False}
total_rewards = {agent_id: 0 for agent_id in obs.keys()}
portfolio_history = {agent_id: [] for agent_id in obs.keys()}

# Debug: contadores de ações
action_counts = {0: 0, 1: 0, 2: 0}

# === Inicializa os estados LSTM ===
policies = algo.workers.local_worker().policy_map
agent_states = {
    agent_id: policies[policy_mapping_fn(agent_id, None, None)].get_initial_state()
    for agent_id in obs
}

# Loop de avaliação
step_num = 0
done = {"__all__": False}

while not done["__all__"]:
    print(f"\n🟢 Step {step_num}")
    for agent_id in obs:
        print(f"  Obs {agent_id}: {obs[agent_id]}")

    # Prepara inputs para compute_actions
    actions, new_states, _ = algo.compute_actions(
        observations={agent_id: obs[agent_id] for agent_id in obs},
        state={agent_id: agent_states[agent_id] for agent_id in obs},
        policy_id="shared_policy",  # Assumindo política única
        explore=True
    )


    print(f"  Ações tomadas: {actions}")
    for a in actions.values():
        action_counts[a] += 1

    # Executa passo no ambiente
    obs, rewards, terminated, truncated, info = env.step(actions)
    done = {"__all__": terminated["__all__"] or truncated["__all__"]}

    print(f"  Recompensas: {rewards}")
    print(f"  Posições: {env.positions}")
    print(f"  Caixa: {env.cash:.2f}")

    for agent_id in rewards:
        total_rewards[agent_id] += rewards[agent_id]
        portfolio_value = info[agent_id].get("portfolio_value", np.nan)
        portfolio_history[agent_id].append(portfolio_value)

        # Atualiza o estado do agente
        agent_states[agent_id] = new_states[agent_id]

    step_num += 1

print("\n🔍 Contagem de ações durante o episódio:")
for a, count in action_counts.items():
    action_name = {0: "Hold", 1: "Buy", 2: "Sell"}.get(a, str(a))
    print(f"  {action_name} ({a}): {count} vezes")


# === 7. Resultados finais ===
result_data = []
print("\n🎯 Avaliação Final:")
for agent_id, reward in total_rewards.items():
    values = portfolio_history[agent_id]
    initial_value = values[0]
    final_value = values[-1]
    pct_return = (final_value - initial_value) / initial_value
    avg_return_per_step = np.mean(np.diff(values)) if len(values) > 1 else 0.0

    result_data.append({
        "agent_id": agent_id,
        "total_reward": reward,
        "initial_portfolio": initial_value,
        "final_portfolio": final_value,
        "pct_return": pct_return,
        "avg_return_per_step": avg_return_per_step
    })

    print(f"Agente {agent_id}:")
    print(f"  • Recompensa total     = {reward:.2f}")
    print(f"  • Valor inicial        = {initial_value:.2f}")
    print(f"  • Valor final          = {final_value:.2f}")
    print(f"  • Retorno (%)          = {pct_return * 100:.2f}%")
    print(f"  • Retorno médio/step   = {avg_return_per_step:.2f}")

# === Linha extra: portfólio total (todos os agentes)
total_values = env.portfolio_total_history
initial_total = total_values[0]
final_total = total_values[-1]
pct_total_return = (final_total - initial_total) / initial_total
avg_total_return_per_step = np.mean(np.diff(total_values)) if len(total_values) > 1 else 0.0

result_data.append({
    "agent_id": "TOTAL",
    "total_reward": np.sum(list(total_rewards.values())),
    "initial_portfolio": initial_total,
    "final_portfolio": final_total,
    "pct_return": pct_total_return,
    "avg_return_per_step": avg_total_return_per_step
})

print(f"\n📊 Portfólio Total:")
print(f"  • Valor inicial        = {initial_total:.2f}")
print(f"  • Valor final          = {final_total:.2f}")
print(f"  • Retorno (%)          = {pct_total_return * 100:.2f}%")
print(f"  • Retorno médio/step   = {avg_total_return_per_step:.2f}")

# Salva CSV com todos os dados
results_df = pd.DataFrame(result_data)
results_csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\n📁 Resultados salvos em: {results_csv_path}")

# Salva CSV da série temporal do portfólio total
total_df = pd.DataFrame({
    "time_step": list(range(len(total_values))),
    "portfolio_total": total_values
})
total_series_path = os.path.join(output_dir, "portfolio_total_series.csv")
total_df.to_csv(total_series_path, index=False)
print(f"📁 Série temporal do portfólio total salva em: {total_series_path}")

# === 8. Gráfico da evolução do portfólio ===
plt.figure(figsize=(12, 6))
for agent_id, values in portfolio_history.items():
    plt.plot(values, label=f"{agent_id}")
plt.title("📈 Evolução do Valor do Portfólio por Agente")
plt.xlabel("Time Step")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "portfolio_evolution.png"))
print(f"📊 Gráfico salvo em {output_dir}/portfolio_evolution.png")

# Salva evolução total do portfólio
total_portfolio_series = env.portfolio_total_history
plt.figure(figsize=(10, 5))
plt.plot(total_portfolio_series, label="Portfólio Total (todos os agentes)")
plt.title("📊 Evolução do Portfólio Total")
plt.xlabel("Time Step")
plt.ylabel("Valor do Portfólio")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "portfolio_total.png"))
print(f"📊 Gráfico do portfólio total salvo em {output_dir}/portfolio_total.png")
