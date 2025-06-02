import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class SharedLSTMModel(TorchModelV2, nn.Module):
    """
    Shared LSTM model for multi-agent trading (Pham et al., 2021).

    Arquitetura:
    - FC com 256 unidades + tanh
    - LSTM com 256 unidades (sequência temporal T > 1)
    - Heads lineares separadas: policy (logits) e value (critic)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = int(np.product(obs_space.shape))  # Esperado: 3
        self.num_outputs = num_outputs
        self.lstm_hidden_size = model_config.get("lstm_cell_size", 256)

        # FC camada densa (256 unidades + tanh)
        self.fc = SlimFC(
            in_size=self.obs_size,
            out_size=256,
            activation_fn="tanh"
        )

        # LSTM com batch_first=True (espera entrada como [B, T, F])
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden_size,
            batch_first=True
        )

        # Heads separadas
        self.policy_head = SlimFC(
            in_size=self.lstm_hidden_size,
            out_size=self.num_outputs,
            activation_fn=None
        )

        self.value_head = SlimFC(
            in_size=self.lstm_hidden_size,
            out_size=1,
            activation_fn=None
        )

        self._value_out = None

    @override(TorchModelV2)
    def get_initial_state(self):
        # Inicializa estado escondido (h, c) do LSTM como zeros
        return [
            torch.zeros(self.lstm_hidden_size),
            torch.zeros(self.lstm_hidden_size)
        ]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Entrada: 
        - input_dict["obs"]: (B, T, obs_size) já vem formatado pelo RLlib
        - state: [h, c] do LSTM
        - seq_lens: sequência válida por batch
        """
        x = input_dict["obs"].float()  # (B, T, obs_dim)

        # Aplica FC frame a frame (pode aplicar em todo o tensor)
        B, T, _ = x.shape
        x = x.reshape(B * T, self.obs_size)
        x = self.fc(x)  # (B*T, 256)
        x = x.reshape(B, T, -1)  # (B, T, 256)

        # Ajusta estados para shape esperado do LSTM: (1, B, H)
        h_in = state[0].unsqueeze(0)
        c_in = state[1].unsqueeze(0)

        # LSTM processa sequência (B, T, 256)
        lstm_out, (h_n, c_n) = self.lstm(x, (h_in, c_in))  # lstm_out: (B, T, H)

        # Último output da sequência para cada batch
        final_outputs = lstm_out[:, -1, :]  # (B, H)

        # Heads
        logits = self.policy_head(final_outputs)  # (B, num_outputs)
        self._value_out = self.value_head(final_outputs).squeeze(1)  # (B,)

        return logits, [h_n.squeeze(0), c_n.squeeze(0)]

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out
