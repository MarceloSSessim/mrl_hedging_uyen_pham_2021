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

    Architecture:
    - FC with 256 units + tanh
    - LSTM with 256 units
    - Separate heads: policy (logits) and value (critic)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = int(np.prod(obs_space.shape))
        self.num_outputs = num_outputs
        self.lstm_hidden_size = model_config.get("lstm_cell_size", 256)

        # Fully connected preprocessing layer
        self.fc = SlimFC(
            in_size=self.obs_size,
            out_size=256,
            activation_fn="tanh"
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden_size,
            batch_first=True
        )

        # Policy head (action logits)
        self.policy_head = SlimFC(
            in_size=self.lstm_hidden_size,
            out_size=self.num_outputs,
            activation_fn=None
        )

        # Value head
        self.value_head = SlimFC(
            in_size=self.lstm_hidden_size,
            out_size=1,
            activation_fn=None
        )

        self._value_out = None

    @override(TorchModelV2)
    def get_initial_state(self):
        # Return (B, H) states; RLlib will expand for batch internally
        return [
            torch.zeros(self.lstm_hidden_size, dtype=torch.float32),
            torch.zeros(self.lstm_hidden_size, dtype=torch.float32),
        ]

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, F) → (B, 1, F)

        B, T, _ = x.shape

        # Pass through FC layer
        x = x.reshape(B * T, self.obs_size)
        x = self.fc(x)
        x = x.reshape(B, T, -1)

        device = x.device

        # Prepare LSTM hidden state
        device = x.device

        if len(state) == 0 or state[0].shape[0] != x.shape[0]:
            B = x.shape[0]
            h_in = torch.zeros(1, B, self.lstm_hidden_size, device=device)
            c_in = torch.zeros(1, B, self.lstm_hidden_size, device=device)
        else:
            h_in = state[0].unsqueeze(0).to(device)
            c_in = state[1].unsqueeze(0).to(device)



        lstm_out, (h_n, c_n) = self.lstm(x, (h_in, c_in))

        # Use the last output of the LSTM for policy and value heads
        final_output = lstm_out[:, -1, :]  # (B, H)

        # Compute action logits and value
        logits = self.policy_head(final_output)
        self._value_out = self.value_head(final_output).squeeze(1)

        return logits, [
            h_n.squeeze(0).detach().cpu(),  # (1, B, H) → (B, H)
            c_n.squeeze(0).detach().cpu()
        ]

    @override(TorchModelV2)
    def value_function(self):
        assert self._value_out is not None, "value_function called before forward pass"
        return self._value_out

