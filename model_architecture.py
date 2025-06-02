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
    - LSTM com 256 unidades
    - Heads separadas: policy (logits) e value (critic)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = int(np.prod(obs_space.shape))
        self.num_outputs = num_outputs
        self.lstm_hidden_size = model_config.get("lstm_cell_size", 256)

        self.fc = SlimFC(
            in_size=self.obs_size,
            out_size=256,
            activation_fn="tanh"
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden_size,
            batch_first=True
        )

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
        h = torch.zeros(self.lstm_hidden_size, dtype=torch.float32)
        c = torch.zeros(self.lstm_hidden_size, dtype=torch.float32)
        return [h, c]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Args:
            input_dict["obs"]: Tensor de observações com shape (B, T, obs_size) ou (B, obs_size)
            state: lista com dois tensores [h, c] representando os estados oculto e de célula da LSTM
            seq_lens: tamanho efetivo das sequências (usado pelo RLlib)

        Returns:
            logits: ações da política (B, num_outputs)
            new_state: lista com novos estados [h_n, c_n]
        """

        # --- Entrada: observação ---
        x = input_dict["obs"].float()
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, F) → (B, 1, F)

        B, T, _ = x.shape

        x = x.reshape(B * T, self.obs_size)
        x = self.fc(x)
        x = x.reshape(B, T, -1)

        device = x.device

        if len(state) == 0:
            h_in = torch.zeros(1, B, self.lstm_hidden_size, device=device)
            c_in = torch.zeros(1, B, self.lstm_hidden_size, device=device)
        else:
            h_in = state[0].unsqueeze(0).to(device)  # (1, B, H)
            c_in = state[1].unsqueeze(0).to(device)

        lstm_out, (h_n, c_n) = self.lstm(x, (h_in, c_in))

        final_outputs = lstm_out[:, -1, :]
        logits = self.policy_head(final_outputs)
        self._value_out = self.value_head(final_outputs).squeeze(1)

        return logits, [
            h_n.squeeze(0).detach().cpu(),
            c_n.squeeze(0).detach().cpu()
        ]


    @override(TorchModelV2)
    def value_function(self):
        assert self._value_out is not None, "value_function called before forward pass"
        return self._value_out

