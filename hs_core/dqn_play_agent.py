import torch
from rl.dqn_agent import DQNAgent
from rl.state_encoder import encode_state


class DQNPlayAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.agent = DQNAgent(state_dim, action_dim)
        self.agent.policy_net.load_state_dict(
            torch.load(
                model_path,
                map_location="mps" if torch.backends.mps.is_available() else "cpu",
            )
        )
        self.agent.policy_net.eval()
        # Set epsilon to 0 for pure exploitation during evaluation
        self.agent.epsilon = 0.0
        self.agent.epsilon_end = 0.0
        self.agent.epsilon_decay = 1.0

    def choose_action(self, state):
        legal_actions = state.get_legal_actions()
        action_map = {i: a for i, a in enumerate(legal_actions)}
        state_vec = encode_state(state, state.current)
        action_idx = self.agent.select_action(state_vec, list(action_map.keys()))
        return action_map[action_idx]
