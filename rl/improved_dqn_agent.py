from rl.dqn_agent import DQNAgent
import torch
import torch.nn as nn
from rl.state_encoder import encode_state
from rl.utils import get_learning_rate
import rl.constants as constants


class ImprovedDQNAgent(DQNAgent):
    """Enhanced DQN agent with improved network architecture"""

    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)

        # Replace the network with a deeper one
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
        ).to(self.device)

        self.target_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=constants.LR)

    def update_learning_rate(self, episode):
        """Update learning rate according to schedule"""
        new_lr = get_learning_rate(episode)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
