import torch
import torch.nn as nn
import numpy as np
from rl.utils import enhanced_encode_state


class ImprovedDQN(nn.Module):
    """Same network architecture as used in training"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.policy_net(x)


class DQNPlayAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.policy_net = ImprovedDQN(state_dim, action_dim)
        # Load the trained model
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # Load weights into the inner nn.Sequential, not the wrapper module
        self.policy_net.policy_net.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        self.device = device
        self.policy_net = self.policy_net.to(self.device)
        self.policy_net.eval()

    def choose_action(self, state):
        legal_actions = state.get_legal_actions()
        action_map = {i: a for i, a in enumerate(legal_actions)}

        # Use the same enhanced state encoder as training
        state_vec = enhanced_encode_state(state, state.current)

        # Convert to tensor and get Q-values
        with torch.no_grad():
            state_tensor = (
                torch.tensor(state_vec, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )
            q_values = self.policy_net(state_tensor)

            # Filter Q-values for legal actions only
            legal_action_indices = list(action_map.keys())
            legal_q_values = q_values[0, legal_action_indices]

            # Select action with highest Q-value
            best_action_idx = legal_action_indices[torch.argmax(legal_q_values).item()]

        return action_map[best_action_idx]
