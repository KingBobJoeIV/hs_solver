# Require a minimum number of episodes before early stopping
MIN_TRAIN_EPISODES = 20000
# MIN_TRAIN_EPISODES = 0
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
    # Stronger incentives for lethal, attacking, coin, and mana use
    "hero_damage": 0.15,  # Slightly increased
    "self_damage": -0.2,
    "board_presence": 0.45,  # Increased
    "kill_minion": 0.30,
    "lose_minion": -0.20,
    "overdraw": -0.3,
    "fatigue": -0.4,
    "mana_waste": -0.3,  # Stronger penalty
    "mana_efficiency": 0.3,  # Stronger bonus
    "win_reward": 12.0,  # Even stronger terminal reward
    "loss_penalty": -12.0,
    "turn_length_penalty": -0.01,
    # Coin usage
    "coin_usage": 0.10,  # Small bonus for any use
    "coin_play_minion": 1.0,  # Large bonus for above-curve minion
    "coin_board_swing": 0.70,  # Larger for board swing
    "coin_removal": 0.8,  # Larger for removal
    "coin_waste": -0.5,  # Penalty for wasting coin
    # Attacking and lethal
    "lethal_bonus": 2.0,  # Bonus for delivering lethal
    "attack_bonus": 0.2,  # Bonus for each attack
    "skip_attack_penalty": -0.2,  # Penalty for not attacking with available minion
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
