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
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility

from env_trading import MultiAgentTradingEnv
from model_architecture import SharedLSTMModel

# === 0. Verifica se há GPU disponível ===
print("✅ PyTorch CUDA disponível:", torch.cuda.is_available())

# === 1. Inicializa Ray ===
ray.init(ignore_reinit_error=True)

# === 2. Registra modelo customizado ===
ModelCatalog.register_custom_model("shared_lstm_model", SharedLSTMModel)

# === 3. Define e registra ambiente customizado ===
def create_env(env_config):
    assert os.path.exists(env_config["price_path"]), f"Arquivo não encontrado: {env_config['price_path']}"
    assert os.path.exists(env_config["return_path"]), f"Arquivo não encontrado: {env_config['return_path']}"

    price_df = pd.read_csv(env_config["price_path"], index_col=0).astype(np.float32).iloc[:100]
    return_df = pd.read_csv(env_config["return_path"], index_col=0).astype(np.float32).iloc[:100]
    asset_types = env_config["asset_types"]

    return MultiAgentTradingEnv(
        price_df=price_df,
        log_return_df=return_df,
        asset_types=asset_types,
        initial_cash=env_config.get("initial_cash", 1e6),
        transaction_fee=env_config.get("transaction_fee", 0.001),
        future_discount=env_config.get("future_discount", 0.001),
    )

   # return MultiAgentEnvCompatibility(env)

register_env("MultiAgentTradingEnv-v0", create_env)

# === 4. Caminhos e configurações do ambiente ===
price_path = os.path.join("data", "processed", "raw_prices.csv")
return_path = os.path.join("data", "processed", "returns_log.csv")
asset_types = ["equity"] * 10 + ["future"]
abs_results_path = os.path.abspath("results")

# === 5. Política compartilhada para todos os agentes ===
def policy_mapping_fn(agent_id, episode, **kwargs):
    return "shared_policy"

# === 6. Espaços de observação e ação ===
temp_env = create_env({
    "price_path": price_path,
    "return_path": return_path,
    "asset_types": asset_types,
})
obs_space = temp_env.observation_space
act_space = temp_env.action_space

# === 7. Configuração do IMPALA ===
config = ImpalaConfig()

# Configuração do ambiente
config = config.environment(
    env="MultiAgentTradingEnv-v0",
    env_config={
        "price_path": price_path,
        "return_path": return_path,
        "asset_types": asset_types,
        "initial_cash": 1e6,
        "transaction_fee": 0.001,
        "future_discount": 0.001,
    },
    disable_env_checking=True  # 👈 ADICIONE ISSO
)

# Configuração do backend
config = config.framework("torch")

# Recursos
config = config.resources(
    num_gpus=1,
    num_cpus_per_worker=1
)

# Rollouts
config = config.rollouts(
    num_rollout_workers=6,
    rollout_fragment_length=20
)

# Treinamento
config = config.training(
    lr=3e-4,
    train_batch_size=1280,
    vf_loss_coeff=0.5,
    entropy_coeff=0.01,
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
config.model["max_seq_len"] = 20

# Parâmetro experimental
config = config.experimental(_disable_preprocessor_api=True)

# === 8. Executa o treinamento ===
analysis = Tuner(
    "IMPALA",
    run_config=RunConfig(
        stop={"training_iteration": 2},
        local_dir=abs_results_path,
        name="impala_trading_experiment",
        log_to_file=True,
        checkpoint_config=CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_frequency=10
        ),
        verbose=1
    ),
    param_space=config,
).fit()

# === 9. Salva resultados ===
with open(os.path.join("results", "impala_analysis.pkl"), "wb") as f:
    pickle.dump(analysis, f)

print("✅ Treinamento finalizado.")

# === 10. Estatísticas finais ===
df = analysis.get_dataframe()
try:
    print(df[["training_iteration", "episode_reward_mean", "episode_len_mean"]].tail())
except KeyError:
    print("🚫 Nenhum resultado disponível. O treinamento falhou.")

# === 11. Melhor checkpoint ===
best_result = analysis.get_best_result()
best_checkpoint = best_result.checkpoint

print(f"🏁 Melhor checkpoint salvo em: {best_checkpoint}")

# Salvar DataFrame diretamente como CSV
df.to_csv("./results/impala_training_metrics.csv", index=False)
