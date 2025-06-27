import torch
import torch.nn as nn
import numpy as np
from rl.dqn_agent import DQNAgent
from rl.state_encoder import encode_state
from toy_hearthstone import setup_game, describe_action, print_board
from hs_core.random_agent import RandomAgent
import random
import matplotlib.pyplot as plt
import copy
import os
import math
import time
from collections import deque

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
    "hero_damage": 0.3,
    "self_damage": -0.2,
    "board_presence": 0.15,
    "kill_minion": 0.2,
    "lose_minion": -0.15,
    "overdraw": -0.3,
    "fatigue": -0.4,
    "mana_waste": -0.08,
    "mana_efficiency": 0.03,
    "win_reward": 3.0,
    "loss_penalty": -3.0,
    "turn_length_penalty": -0.01,
}

# Curriculum and early stopping
CURRICULUM_THRESHOLD = 0.6
EARLY_STOP_PATIENCE = 5
MIN_IMPROVEMENT = 0.02
TARGET_WIN_RATE = 0.90
WARMUP_EPISODES = 1000

# Action space size
ACTION_SPACE_SIZE = 50


def get_beta(episode):
    """Beta annealing for prioritized replay"""
    return min(BETA_END, BETA_START + (BETA_END - BETA_START) * episode / EPISODES)


def get_learning_rate(episode):
    """Cosine annealing learning rate schedule"""
    return LR * 0.5 * (1 + math.cos(math.pi * episode / EPISODES))


def enhanced_encode_state(game_state, player_idx):
    """Enhanced state encoding with normalized features"""
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


def enhanced_calculate_reward(game, prev_game_state, action, acting_player_idx):
    """Enhanced reward function with better strategic understanding"""
    reward = 0.0

    current_player = game.players[acting_player_idx]
    opponent = game.players[1 - acting_player_idx]
    prev_current_player = prev_game_state.players[acting_player_idx]
    prev_opponent = prev_game_state.players[1 - acting_player_idx]

    # Hero damage (primary objective)
    hero_damage = prev_opponent.hero_hp - opponent.hero_hp
    if hero_damage > 0:
        reward += REWARD_CONFIG["hero_damage"] * hero_damage

    # Self damage penalty
    self_damage = prev_current_player.hero_hp - current_player.hero_hp
    if self_damage > 0:
        reward += REWARD_CONFIG["self_damage"] * self_damage

    # Board control
    board_increase = len(current_player.board) - len(prev_current_player.board)
    if board_increase > 0:
        reward += REWARD_CONFIG["board_presence"] * board_increase

    # Minion trading evaluation
    prev_opp_minions = {id(m): m for m in prev_opponent.board}
    curr_opp_minions = {id(m): m for m in opponent.board}

    for minion_id, minion in prev_opp_minions.items():
        if minion_id not in curr_opp_minions:
            reward += (
                REWARD_CONFIG["kill_minion"] * (minion.attack + minion.health) / 6.0
            )

    prev_own_minions = {id(m): m for m in prev_current_player.board}
    curr_own_minions = {id(m): m for m in current_player.board}

    for minion_id, minion in prev_own_minions.items():
        if minion_id not in curr_own_minions:
            reward += (
                REWARD_CONFIG["lose_minion"] * (minion.attack + minion.health) / 6.0
            )

    # Resource management penalties
    deck_decrease = len(prev_current_player.deck) - len(current_player.deck)
    hand_increase = len(current_player.hand) - len(prev_current_player.hand)

    if deck_decrease > 0 and hand_increase == 0 and len(prev_current_player.hand) >= 4:
        reward += REWARD_CONFIG["overdraw"]

    fatigue_damage = current_player.fatigue - prev_current_player.fatigue
    if fatigue_damage > 0:
        reward += REWARD_CONFIG["fatigue"] * fatigue_damage

    # Mana efficiency
    if action[0] == "end":
        wasted_mana = prev_current_player.mana
        if wasted_mana > 1:
            reward += REWARD_CONFIG["mana_waste"] * wasted_mana

    mana_spent = prev_current_player.mana - current_player.mana
    if mana_spent > 0 and action[0] == "play":
        reward += REWARD_CONFIG["mana_efficiency"] * mana_spent

    # Terminal rewards (most important)
    if game.is_terminal():
        if game.winner == acting_player_idx:
            reward += REWARD_CONFIG["win_reward"]
        elif game.winner is not None:
            reward += REWARD_CONFIG["loss_penalty"]

    return reward


def get_action_index_map(game, legal_actions):
    """Map legal actions to indices for DQN"""
    action_map = {i: a for i, a in enumerate(legal_actions)}
    return action_map, {str(a): i for i, a in enumerate(legal_actions)}


def evaluate_agent(agent, opponent_type="random", num_games=200, verbose=False):
    """Evaluate agent performance"""
    wins = 0

    for _ in range(num_games):
        if opponent_type == "random":
            opponent = RandomAgent()
        else:
            opponent = opponent_type

        # Random first player assignment
        agents = [agent, opponent]
        random.shuffle(agents)
        agent_idx = 0 if agents[0] is agent else 1

        game = setup_game()
        game.players[0].start_turn()

        while not game.is_terminal():
            current_idx = game.current
            acting_agent = agents[current_idx]

            legal = game.get_legal_actions()
            amap, _ = get_action_index_map(game, legal)
            state = enhanced_encode_state(game, current_idx)

            if acting_agent is agent:
                # Use greedy policy for evaluation
                old_epsilon = agent.epsilon
                agent.epsilon = 0.0
                action_idx = agent.select_action(state, list(amap.keys()))
                agent.epsilon = old_epsilon
            else:
                if hasattr(acting_agent, "select_action"):
                    try:
                        with torch.no_grad():
                            acting_agent.epsilon = 0.0
                            action_idx = acting_agent.select_action(
                                state, list(amap.keys())
                            )
                    except:
                        action_idx = random.choice(list(amap.keys()))
                else:
                    action_idx = random.choice(list(amap.keys()))

            action = amap[action_idx]
            game.step(action)

        if game.winner == agent_idx:
            wins += 1

    win_rate = wins / num_games
    if verbose:
        print(f"Win rate vs {opponent_type}: {win_rate:.3f} ({wins}/{num_games})")

    return win_rate


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
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)

    def update_learning_rate(self, episode):
        """Update learning rate according to schedule"""
        new_lr = get_learning_rate(episode)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


def improved_dqn_training():
    """Main training loop with all improvements"""

    # Create directories
    os.makedirs("dqn_policy", exist_ok=True)
    os.makedirs("dqn_logs", exist_ok=True)

    # Get state dimensions
    dummy_game = setup_game()
    dummy_state = enhanced_encode_state(dummy_game, 0)
    STATE_SIZE = dummy_state.shape[0]

    print(f"State size: {STATE_SIZE}")
    print(f"Action space size: {ACTION_SPACE_SIZE}")
    print(f"Training for {EPISODES} episodes")

    # Initialize agents
    agent = ImprovedDQNAgent(
        STATE_SIZE,
        ACTION_SPACE_SIZE,
        lr=LR,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        prioritized_replay=True,
        alpha=ALPHA,
        beta=BETA_START,
    )

    target_agent = ImprovedDQNAgent(
        STATE_SIZE,
        ACTION_SPACE_SIZE,
        lr=LR,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        epsilon_decay=EPSILON_DECAY,
        prioritized_replay=True,
    )
    target_agent.policy_net.load_state_dict(agent.policy_net.state_dict())

    # Training tracking
    win_history = []
    baseline_history = []
    loss_history = []
    epsilon_history = []
    best_win_rate = 0.0
    curriculum_phase = 0
    early_stop_counter = 0

    # Training statistics
    episode_rewards = deque(maxlen=1000)
    episode_lengths = deque(maxlen=1000)

    start_time = time.time()

    print("Starting training...")
    print(f"Epsilon decay rate: {EPSILON_DECAY:.6f}")
    print("-" * 50)

    for episode in range(1, EPISODES + 1):
        # Curriculum opponent selection
        if curriculum_phase == 0:
            opponent = RandomAgent()
        else:
            if random.random() < 0.2:  # 20% random opponents
                opponent = RandomAgent()
            else:
                opponent = target_agent

        # Random first player assignment
        agents = [agent, opponent]
        random.shuffle(agents)
        agent_player_idx = 0 if agents[0] is agent else 1

        game = setup_game()
        game.players[0].start_turn()

        episode_reward = 0.0
        episode_length = 0

        # Game loop
        while not game.is_terminal() and episode_length < 100:  # Prevent infinite games
            current_player_idx = game.current
            acting_agent = agents[current_player_idx]

            state = enhanced_encode_state(game, current_player_idx)
            legal_actions = game.get_legal_actions()
            action_map, _ = get_action_index_map(game, legal_actions)

            # Choose action
            if acting_agent is agent:
                action_idx = agent.select_action(state, list(action_map.keys()))
            else:
                if hasattr(acting_agent, "select_action"):
                    try:
                        with torch.no_grad():
                            action_idx = acting_agent.select_action(
                                state, list(action_map.keys())
                            )
                    except:
                        action_idx = random.choice(list(action_map.keys()))
                else:
                    action_idx = random.choice(list(action_map.keys()))

            action = action_map[action_idx]
            prev_game_state = copy.deepcopy(game)

            # Execute action
            game.step(action)
            episode_length += 1

            # Store experience and update (only for our training agent)
            if acting_agent is agent and episode > WARMUP_EPISODES:
                next_state = enhanced_encode_state(game, game.current)
                reward = enhanced_calculate_reward(
                    game, prev_game_state, action, current_player_idx
                )

                # Add turn length penalty for very long games
                if episode_length > 50:
                    reward += REWARD_CONFIG["turn_length_penalty"]

                done = game.is_terminal()
                episode_reward += reward

                # Update beta for prioritized replay
                agent.beta = get_beta(episode)

                agent.store_transition(state, action_idx, reward, next_state, done)
                agent.update()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Update learning rate
        if episode > WARMUP_EPISODES:
            agent.update_learning_rate(episode)

        # Target network update
        if episode % TARGET_UPDATE == 0:
            agent.update_target()
            target_agent.policy_net.load_state_dict(agent.policy_net.state_dict())

        # Logging
        if episode % 100 == 0 or episode == 1:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            print(
                f"Episode {episode:5d} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Avg Reward: {avg_reward:6.2f} | "
                f"Avg Length: {avg_length:4.1f} | "
                f"Phase: {curriculum_phase}"
            )

        # Evaluation
        if episode % EVAL_INTERVAL == 0 and episode > WARMUP_EPISODES:
            # Evaluate against random agent
            baseline_win_rate = evaluate_agent(agent, "random", num_games=200)
            baseline_history.append(baseline_win_rate)

            # Self-play evaluation
            if curriculum_phase > 0:
                self_play_win_rate = evaluate_agent(agent, target_agent, num_games=100)
                win_history.append(self_play_win_rate)
            else:
                win_history.append(0.5)  # Placeholder during phase 0

            epsilon_history.append(agent.epsilon)

            print(f"\nEvaluation at episode {episode}:")
            print(f"  Win rate vs Random: {baseline_win_rate:.3f}")
            if curriculum_phase > 0:
                print(f"  Win rate vs Self:   {self_play_win_rate:.3f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Learning rate: {get_learning_rate(episode):.6f}")
            print("-" * 50)

            # Curriculum advancement
            if curriculum_phase == 0 and baseline_win_rate > CURRICULUM_THRESHOLD:
                print(f"🎓 Advancing to curriculum phase 1 at episode {episode}!")
                curriculum_phase = 1

            # Save best model
            if baseline_win_rate > best_win_rate:
                improvement = baseline_win_rate - best_win_rate
                best_win_rate = baseline_win_rate
                torch.save(
                    agent.policy_net.state_dict(),
                    f"dqn_policy/dqn_policy_best_ep{episode}.pt",
                )
                print(
                    f"💾 New best model saved! Win rate: {baseline_win_rate:.3f} (+{improvement:.3f})"
                )
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Early stopping
            if baseline_win_rate >= TARGET_WIN_RATE:
                print(
                    f"🎯 Target win rate {TARGET_WIN_RATE:.2f} achieved! Stopping early."
                )
                break

            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(
                    f"⏹️  Early stopping: No improvement for {EARLY_STOP_PATIENCE} evaluations"
                )
                break

        # Save checkpoints
        if episode % SAVE_INTERVAL == 0:
            torch.save(
                agent.policy_net.state_dict(), f"dqn_policy/dqn_policy_ep{episode}.pt"
            )

    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\n🏁 Training completed in {elapsed_time / 3600:.2f} hours")
    print(f"Best win rate achieved: {best_win_rate:.3f}")

    # Save final model
    torch.save(agent.policy_net.state_dict(), "dqn_policy/dqn_policy_final.pt")

    # Plot results
    if len(baseline_history) > 1:
        episodes_eval = list(
            range(
                EVAL_INTERVAL, EVAL_INTERVAL * len(baseline_history) + 1, EVAL_INTERVAL
            )
        )

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(episodes_eval, baseline_history, "b-", linewidth=2, label="vs Random")
        if len(win_history) > 1:
            plt.plot(episodes_eval, win_history, "r--", linewidth=2, label="vs Self")
        plt.axhline(
            y=TARGET_WIN_RATE,
            color="g",
            linestyle=":",
            label=f"Target ({TARGET_WIN_RATE})",
        )
        plt.xlabel("Episode")
        plt.ylabel("Win Rate")
        plt.title("Win Rate Progress")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.plot(episodes_eval, epsilon_history, "purple", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Exploration Rate")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        recent_rewards = list(episode_rewards)[-1000:]
        if recent_rewards:
            plt.plot(recent_rewards, alpha=0.6)
            # Moving average
            window = 100
            if len(recent_rewards) > window:
                moving_avg = [
                    np.mean(recent_rewards[i : i + window])
                    for i in range(len(recent_rewards) - window)
                ]
                plt.plot(
                    range(window, len(recent_rewards)), moving_avg, "r-", linewidth=2
                )
        plt.xlabel("Recent Episodes")
        plt.ylabel("Episode Reward")
        plt.title("Reward Progress")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("dqn_policy/training_progress.png", dpi=300, bbox_inches="tight")
        plt.show()

    return agent, best_win_rate


if __name__ == "__main__":
    print("🚀 Starting improved DQN training...")
    agent, final_win_rate = improved_dqn_training()
    print(f"✅ Training finished with final win rate: {final_win_rate:.3f}")
