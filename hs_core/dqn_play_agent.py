import torch
import torch.nn as nn
import numpy as np


def enhanced_encode_state(game_state, player_idx):
    """Enhanced state encoding - same as used in training"""
    player = game_state.players[player_idx]
    opponent = game_state.players[1 - player_idx]
    state = []

    # Basic player info (normalized)
    state += [
        player.hero_hp / 5.0,
        player.mana / 5.0,
        player.max_mana / 5.0,
        len(player.deck) / 7.0,
        player.fatigue / 5.0,
        opponent.hero_hp / 5.0,
        opponent.mana / 5.0,
        opponent.max_mana / 5.0,
        len(opponent.deck) / 7.0,
        opponent.fatigue / 5.0,
    ]

    # Turn information
    state += [
        player.turn / 10.0,
        int(player_idx == game_state.current),
    ]

    # Enhanced hand encoding
    card_features = {
        "1/1": [1 / 5, 1 / 5, 1 / 5, 0],
        "1 mana deal 1": [1 / 5, 1 / 5, 0, 1],
        "2/2": [2 / 5, 2 / 5, 2 / 5, 0],
        "2 1/1 deal 1": [2 / 5, 1 / 5, 1 / 5, 0],
        "3/3": [3 / 5, 3 / 5, 3 / 5, 0],
        "3 2/2 deal 1": [3 / 5, 2 / 5, 2 / 5, 0],
        "3 mana deal 2": [3 / 5, 2 / 5, 0, 1],
        "The Coin": [0, 0, 0, 1],
    }

    for i in range(4):
        if i < len(player.hand):
            card = player.hand[i]
            if card.name in card_features:
                state += card_features[card.name]
            else:
                state += [0, 0, 0, 0]
        else:
            state += [0, 0, 0, 0]

    # Board encoding (normalized)
    for minion in player.board[:3]:
        state += [minion.attack / 5.0, minion.health / 5.0, int(minion.can_attack)]
    for _ in range(3 - len(player.board)):
        state += [0, 0, 0]

    for minion in opponent.board[:3]:
        state += [minion.attack / 5.0, minion.health / 5.0, int(minion.can_attack)]
    for _ in range(3 - len(opponent.board)):
        state += [0, 0, 0]

    # Strategic features
    total_player_board_value = sum(m.attack + m.health for m in player.board)
    total_opponent_board_value = sum(m.attack + m.health for m in opponent.board)

    state += [
        total_player_board_value / 15.0,
        total_opponent_board_value / 15.0,
        len(player.board) / 3.0,
        len(opponent.board) / 3.0,
    ]

    return np.array(state, dtype=np.float32)


class ImprovedDQN(nn.Module):
    """Same network architecture as used in training"""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
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
        return self.net(x)


class DQNPlayAgent:
    def __init__(self, state_dim, action_dim, model_path, use_improved_network=True):
        # Choose the correct network architecture
        if use_improved_network:
            self.policy_net = ImprovedDQN(state_dim, action_dim)
        else:
            # Fallback to basic DQN if needed
            from rl.dqn_agent import DQN

            self.policy_net = DQN(state_dim, action_dim)

        # Load the trained model
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.policy_net.load_state_dict(torch.load(model_path, map_location=device))
        self.policy_net.eval()
        self.device = device

    def choose_action(self, state):
        legal_actions = state.get_legal_actions()
        action_map = {i: a for i, a in enumerate(legal_actions)}

        # Use the same enhanced state encoder as training
        state_vec = enhanced_encode_state(state, state.current)

        # Convert to tensor and get Q-values
        with torch.no_grad():
            state_tensor = torch.tensor(
                state_vec, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.policy_net(state_tensor)

            # Filter Q-values for legal actions only
            legal_action_indices = list(action_map.keys())
            legal_q_values = q_values[0, legal_action_indices]

            # Select action with highest Q-value
            best_action_idx = legal_action_indices[torch.argmax(legal_q_values).item()]

        return action_map[best_action_idx]
