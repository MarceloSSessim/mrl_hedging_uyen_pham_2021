import os
import pickle
import torch
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune import Tuner
from ray.rllib.algorithms.impala import ImpalaConfig
from utils import policy_mapping_fn
from env_trading import MultiAgentTradingEnv
from model_architecture import SharedLSTMModel
import json
from pathlib import Path

# ✅ 0. Verifica se há GPU disponível
print("✅ PyTorch CUDA disponível:", torch.cuda.is_available())

# ✅ 1. Inicializa Ray
ray.init(ignore_reinit_error=True)

# ✅ 2. Registra o modelo customizado
ModelCatalog.register_custom_model("shared_lstm_model", SharedLSTMModel)

# ✅ 3. Define e registra o ambiente customizado
def create_env(env_config):
    assert os.path.exists(env_config["price_path"]), f"Arquivo não encontrado: {env_config['price_path']}"
    assert os.path.exists(env_config["return_path"]), f"Arquivo não encontrado: {env_config['return_path']}"

    price_df = pd.read_csv(env_config["price_path"], index_col=0).astype(np.float32).iloc[:500]
    return_df = pd.read_csv(env_config["return_path"], index_col=0).astype(np.float32).iloc[:500]
    asset_types = env_config["asset_types"]

    env = MultiAgentTradingEnv(
        price_df=price_df,
        log_return_df=return_df,
        asset_types=asset_types,
        initial_cash=env_config.get("initial_cash", 500000),
        transaction_fee=env_config.get("transaction_fee", 0.001),
        future_discount=env_config.get("future_discount", 0.001),
    )

    # Adiciona os atributos exigidos pelo RLlib
    env.observation_spaces = {agent_id: env.single_obs_space for agent_id in env.agent_ids}
    env.action_spaces = {agent_id: env.single_act_space for agent_id in env.agent_ids}

    return env

register_env("MultiAgentTradingEnv-v0", create_env)

# ✅ 4. Caminhos e configurações do ambiente
price_path = os.path.join("data", "processed", "raw_prices.csv")
return_path = os.path.join("data", "processed", "returns_log.csv")
asset_types = ["equity"] * 10 + ["future"]
abs_results_path = os.path.abspath("results")

# ✅ 5. Obtém espaços do ambiente temporário para registrar shared_policy
temp_env = create_env({
    "price_path": price_path,
    "return_path": return_path,
    "asset_types": asset_types,
})
obs_space = temp_env.single_obs_space
act_space = temp_env.single_act_space

# ✅ 6. Configuração do algoritmo IMPALA
config = ImpalaConfig()

# Ambiente
config = config.environment(
    env="MultiAgentTradingEnv-v0",
    env_config={
        "price_path": price_path,
        "return_path": return_path,
        "asset_types": asset_types,
        "initial_cash": 500000,
        "transaction_fee": 0.001,
        "future_discount": 0.001,
    },
    disable_env_checking=True
)

# Backend
config = config.framework("torch")

# Recursos
config = config.resources(
    num_gpus=1,
    num_cpus_per_worker=3
)

# Rollouts
config = config.rollouts(
    num_rollout_workers=6,
    rollout_fragment_length=36,
    num_envs_per_worker=2,
)

# Treinamento
config = config.training(
    lr=3e-4,
    train_batch_size=1008,
    vf_loss_coeff=0.5,
    entropy_coeff=0.02,
)

# Multi-agente
config = config.multi_agent(
    policies={
        "shared_policy": (
            None,
            obs_space,
            act_space,
            {}
        )
    },
    policy_mapping_fn=policy_mapping_fn,
)

# Modelo customizado
config.model["custom_model"] = "shared_lstm_model"
config.model["custom_model_config"] = {"lstm_cell_size": 256}
config.model["max_seq_len"] = 12

# Param experimental
config = config.experimental(_disable_preprocessor_api=True)

# ✅ 7. Executa o treinamento
analysis = Tuner(
    "IMPALA",
    run_config=RunConfig(
        stop={"training_iteration": 100},
        local_dir=abs_results_path,
        name="impala_trading_experiment",
        log_to_file=True,
        checkpoint_config=CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_frequency=25
        ),
        verbose=1
    ),
    param_space=config,
).fit()

print("✅ Treinamento finalizado.")

# ✅ 8. Estatísticas finais
df = analysis.get_dataframe()
try:
    print(df[["training_iteration", "episode_reward_mean", "episode_len_mean"]].tail())
except KeyError:
    print("🚫 Nenhum resultado disponível. O treinamento falhou.")

# ✅ 9. Melhor checkpoint
best_result = analysis.get_best_result()
best_checkpoint = best_result.checkpoint

print(f"🏁 Melhor checkpoint salvo em: {best_checkpoint}")

# ✅ 10. Salva DataFrame como CSV
df.to_csv("./results/impala_training_metrics.csv", index=False)

# ✅ 11. Append do melhor checkpoint em JSON (usando path string)
checkpoint_log_path = Path("./results/best_checkpoints.json")

# Carrega a lista existente (se houver)
if checkpoint_log_path.exists():
    with open(checkpoint_log_path, "r") as f:
        checkpoint_list = json.load(f)
else:
    checkpoint_list = []

# Extrai o caminho do checkpoint
checkpoint_path_str = str(best_checkpoint.path)

# Adiciona o novo caminho, evitando duplicatas
if checkpoint_path_str not in checkpoint_list:
    checkpoint_list.append(checkpoint_path_str)

# Salva novamente como JSON
with open(checkpoint_log_path, "w") as f:
    json.dump(checkpoint_list, f, indent=2)


# ✅ 12. Entropia da política (shared_policy)
import matplotlib.pyplot as plt

entropy_col = "info/learner/shared_policy/learner_stats/entropy"

if entropy_col in df.columns:
    print("\n📈 Últimos valores de entropia da política compartilhada:")
    print(df[["training_iteration", entropy_col]].tail())

    # Plota gráfico da entropia ao longo do tempo
    plt.figure(figsize=(8, 4))
    plt.plot(df["training_iteration"], df[entropy_col])
    plt.title("Entropia da política compartilhada durante o treinamento")
    plt.xlabel("Iteração")
    plt.ylabel("Entropia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./results/policy_entropy_plot.png")  # salva como imagem
    plt.show()
else:
    print("🚫 Coluna de entropia da política compartilhada não encontrada.")
