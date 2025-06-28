# Require a minimum number of episodes before early stopping
# MIN_TRAIN_EPISODES = 20000
MIN_TRAIN_EPISODES = 0
# Enhanced hyperparameters
EPISODES = 50000  # Increased for deeper training
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
    # Further reduce hero damage, increase board control and mana efficiency
    "hero_damage": 0.1,  # Further reduced
    "self_damage": -0.2,
    "board_presence": 0.40,  # Slightly increased
    "kill_minion": 0.25,
    "lose_minion": -0.18,
    "overdraw": -0.3,
    "fatigue": -0.4,
    "mana_waste": -0.15,
    "mana_efficiency": 0.15,  # Slightly increased
    "win_reward": 10.0,  # Stronger terminal reward to dominate shaped rewards
    "loss_penalty": -10.0,
    "turn_length_penalty": -0.01,
    # Bonus for using The Coin when going second
    "coin_usage": 0.30,
    # Bonus for using Coin to play a minion above your mana curve when going second
    "coin_play_minion": 0.60,
    # Bonus for gaining board advantage on the same turn as Coin usage
    "coin_board_swing": 0.50,
    # Bonus for using Coin + removal to kill an enemy minion
    "coin_removal": 0.40,
}

# Curriculum and early stopping
CURRICULUM_THRESHOLD = 0.6
EARLY_STOP_PATIENCE = 5
MIN_IMPROVEMENT = 0.02
TARGET_WIN_RATE = 0.97
WARMUP_EPISODES = 1000

# Action space size
ACTION_SPACE_SIZE = 50

# Exploration schedule
EPSILON_START = 1.0
EPSILON_END = 0.05
# Slow down epsilon decay (decay over 90% of episodes instead of 60%)
EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1 / (0.9 * EPISODES))
