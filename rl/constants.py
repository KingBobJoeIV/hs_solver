# Enhanced hyperparameters
EPISODES = 25000
TARGET_UPDATE = 500
EVAL_INTERVAL = 500
SAVE_INTERVAL = 2500
BATCH_SIZE = 64
GAMMA = 0.95
LR = 1e-3
BUFFER_SIZE = 50000

# Exploration schedule
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1 / (0.6 * EPISODES))

# Prioritized replay parameters
ALPHA = 0.6
BETA_START = 0.4
BETA_END = 1.0

# Reward configuration
REWARD_CONFIG = {
    # Increase board control and mana spending, reduce direct hero damage
    "hero_damage": 0.1,
    "self_damage": -0.2,
    "board_presence": 0.35,
    "kill_minion": 0.25,
    "lose_minion": -0.18,
    "overdraw": -0.3,
    "fatigue": -0.4,
    "mana_waste": -0.15,  # Stronger penalty for wasted mana
    "mana_efficiency": 0.12,  # Stronger reward for spending mana
    "win_reward": 3.0,
    "loss_penalty": -3.0,
    "turn_length_penalty": -0.01,
}

# Curriculum and early stopping
CURRICULUM_THRESHOLD = 0.6
EARLY_STOP_PATIENCE = 5
MIN_IMPROVEMENT = 0.02
TARGET_WIN_RATE = 0.85
WARMUP_EPISODES = 1000

# Action space size
ACTION_SPACE_SIZE = 50
