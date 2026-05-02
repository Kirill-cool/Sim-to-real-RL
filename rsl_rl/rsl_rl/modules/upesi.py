import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import get_activation


def _build_mlp(input_dim, hidden_dims, output_dim, activation_name='elu'):
    activation = get_activation(activation_name)
    if activation is None:
        activation = nn.ELU()
    layers = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(activation.__class__())
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


class DynamicsEncoder(nn.Module):
    def __init__(self, theta_dim, embedding_dim, hidden_dims=(64, 64), activation='elu'):
        super().__init__()
        self.net = _build_mlp(theta_dim, hidden_dims, embedding_dim, activation_name=activation)

    def forward(self, theta_norm):
        return self.net(theta_norm)


class ForwardDynamicsModel(nn.Module):
    def __init__(self, obs_dim, action_dim, embedding_dim, hidden_dims=(256, 256), output_dim=None, activation='elu'):
        super().__init__()
        if output_dim is None:
            output_dim = obs_dim
        input_dim = int(obs_dim) + int(action_dim) + int(embedding_dim)
        self.net = _build_mlp(input_dim, hidden_dims, output_dim, activation_name=activation)

    def forward(self, obs, action, alpha):
        x = torch.cat([obs, action, alpha], dim=-1)
        return self.net(x)


class ThetaDecoder(nn.Module):
    def __init__(self, embedding_dim, theta_dim, hidden_dims=(64, 64), activation='elu'):
        super().__init__()
        self.net = _build_mlp(embedding_dim, hidden_dims, theta_dim, activation_name=activation)

    def forward(self, alpha):
        return self.net(alpha)
