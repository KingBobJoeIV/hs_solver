# Require a minimum number of episodes before early stopping
MIN_TRAIN_EPISODES = 20000
# MIN_TRAIN_EPISODES = 0
# Enhanced hyperparameters

# Sparse reward regime: more training, more exploration, larger buffer, slower learning
EPISODES = 100000  # More episodes for sparse rewards
TARGET_UPDATE = 1000  # Less frequent target update
EVAL_INTERVAL = 1000
SAVE_INTERVAL = 5000
BATCH_SIZE = 128  # Larger batch size
GAMMA = 0.99  # Higher discount for long-term credit assignment
LR = 5e-4  # Lower learning rate for stability
BUFFER_SIZE = 100000  # Larger replay buffer

# Improved exploration schedule - no longer using simple decay
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 0.997  # Even slower decay for more exploration

# Prioritized replay parameters
ALPHA = 0.7  # Slightly higher to prioritize rare terminal transitions
BETA_START = 0.5
BETA_END = 1.0

# Improved reward configuration with better balance
REWARD_CONFIG = {
    # Increased direct damage rewards (primary objective)
    "hero_damage": 0.5,  # Increased from 0.15
    "self_damage": -0.3,  # Stronger penalty from -0.2
    # More strategic board control (reduced emphasis)
    "board_presence": 0.2,  # Reduced from 0.45
    "tempo_swing": 0.4,  # New: reward for big board swings
    # Strategic minion trading with efficiency focus
    "kill_minion": 0.2,  # Reduced base from 0.30
    "efficient_trade": 0.5,  # New: bonus for favorable trades
    "lose_minion": -0.3,  # Stronger penalty from -0.20
    # Resource management
    "overdraw": -0.4,  # Increased penalty from -0.3
    "fatigue": -0.5,  # Increased penalty from -0.4
    # Improved mana efficiency system
    "mana_waste": -0.5,  # Stronger penalty from -0.3
    "mana_efficiency": 0.1,  # Reduced base from 0.3
    "optimal_curve": 0.3,  # New: reward for good curve plays
    # Enhanced terminal rewards
    "win_reward": 30.0,  # Much higher for sparse reward
    "loss_penalty": -30.0,  # Much stronger for sparse reward
    "turn_efficiency": 1.0,  # New: bonus for quick wins
    "turn_length_penalty": -0.02,  # Increased from -0.01
    # Strategic coin usage (keeping your good design but tweaking)
    "coin_usage": 0.10,  # Small bonus for any use
    "coin_play_minion": 1.2,  # Increased from 1.0
    "coin_board_swing": 0.8,  # Increased from 0.70
    "coin_removal": 0.9,  # Increased from 0.8
    "coin_waste": -0.7,  # Stronger penalty from -0.5
    # Enhanced attacking incentives
    "lethal_bonus": 3.0,  # Increased from 2.0
    "attack_bonus": 0.3,  # Increased from 0.2
    "skip_attack_penalty": -0.3,  # Stronger from -0.2
}

# Improved curriculum and early stopping
CURRICULUM_THRESHOLD_PHASE_1 = 0.8
CURRICULUM_THRESHOLD_PHASE_2 = 0.85
EARLY_STOP_PATIENCE = 8  # More patience for slow learning
MIN_IMPROVEMENT = 0.01  # Lower threshold for improvement
TARGET_WIN_RATE = 0.95
WARMUP_EPISODES = 3000  # Longer warmup for sparse rewards

# Action space size
ACTION_SPACE_SIZE = 50

# Exploration schedule parameters for adaptive epsilon
EPSILON_PHASE_1_END = 0.3  # 30% of episodes for initial exploration
EPSILON_PHASE_2_END = 0.7  # 70% of episodes for moderate exploration
EPSILON_RESET_INTERVAL = 5000  # Periodic exploration boosts
EPSILON_RESET_VALUE = 0.3  # Reset epsilon to this value

# Network architecture improvements
HIDDEN_SIZE = 512
DROPOUT_RATE = 0.2
GRADIENT_CLIP_NORM = 1.0

# Target network update strategy
USE_SOFT_UPDATE = True
TAU = 0.005  # Soft update parameter

# Advanced training parameters
LR_DECAY_FACTOR = 0.95
LR_DECAY_INTERVAL = 2000
REPLAY_BUFFER_WARMUP = 1000  # Wait before training starts
