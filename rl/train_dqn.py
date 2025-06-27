import torch
import numpy as np
from rl.dqn_agent import DQNAgent
from rl.state_encoder import encode_state
from toy_hearthstone import setup_game, describe_action, print_board
from hs_core.random_agent import RandomAgent
import random
import matplotlib.pyplot as plt
import optuna
import optuna.visualization

# Best hyperparameters from Optuna
EPISODES = 100000  # Increased for more training
TARGET_UPDATE = 1000  # Less frequent target update for stability
EVAL_INTERVAL = 1000
BATCH_SIZE = 128
GAMMA = 0.99
LR = 5.01e-5  # Lower learning rate for stability
BUFFER_SIZE = 100000  # Larger buffer for more diverse experience
EPSILON_DECAY = 0.99927

# Action space size (max possible unique actions)
ACTION_SPACE_SIZE = 50  # Overprovisioned, mask illegal actions

# State size (from encoder)
DUMMY_GAME = setup_game()
DUMMY_STATE = encode_state(DUMMY_GAME, 0)
STATE_SIZE = DUMMY_STATE.shape[0]


def get_action_index_map(game, legal_actions):
    # Map each legal action to a unique index for DQN
    # For simplicity, use str(action) as key
    action_map = {i: a for i, a in enumerate(legal_actions)}
    reverse_map = {str(a): i for i, a in enumerate(legal_actions)}
    return action_map, reverse_map


def calculate_reward(game, prev_game_state, action, acting_player_idx):
    """
    Calculate reward based on game state changes.
    Fixed to properly handle fatigue, overdraw, and other mechanics.
    """
    reward = 0.0

    # Get current and previous states for both players
    current_player = game.players[acting_player_idx]
    opponent = game.players[1 - acting_player_idx]
    prev_current_player = prev_game_state.players[acting_player_idx]
    prev_opponent = prev_game_state.players[1 - acting_player_idx]

    # 1. Reward for dealing damage to opponent hero
    hero_damage = prev_opponent.hero_hp - opponent.hero_hp
    if hero_damage > 0:
        reward += 0.2 * hero_damage

    # 2. Penalty for taking hero damage
    self_damage = prev_current_player.hero_hp - current_player.hero_hp
    if self_damage > 0:
        reward -= 0.15 * self_damage

    # 3. Reward for playing minions (board presence)
    board_increase = len(current_player.board) - len(prev_current_player.board)
    if board_increase > 0:
        reward += 0.1 * board_increase

    # 4. Reward for killing opponent minions (weighted by their stats)
    prev_opp_minions = {id(m): m for m in prev_opponent.board}
    curr_opp_minions = {id(m): m for m in opponent.board}

    for minion_id, minion in prev_opp_minions.items():
        if minion_id not in curr_opp_minions:
            # Minion was killed
            reward += 0.15 * (minion.attack + minion.health)

    # 5. Penalty for losing own minions
    prev_own_minions = {id(m): m for m in prev_current_player.board}
    curr_own_minions = {id(m): m for m in current_player.board}

    for minion_id, minion in prev_own_minions.items():
        if minion_id not in curr_own_minions:
            # Own minion was killed
            reward -= 0.1 * (minion.attack + minion.health)

    # 6. FIXED: Penalty for overdraw (hand was full and card was burned)
    # This happens when deck decreases but hand size doesn't increase to full capacity
    deck_decrease = len(prev_current_player.deck) - len(current_player.deck)
    hand_increase = len(current_player.hand) - len(prev_current_player.hand)

    if deck_decrease > 0 and hand_increase == 0 and len(prev_current_player.hand) >= 4:
        # Card was burned due to full hand
        reward -= 0.2

    # 7. FIXED: Penalty for fatigue damage
    fatigue_damage = current_player.fatigue - prev_current_player.fatigue
    if fatigue_damage > 0:
        # Player took fatigue damage
        reward -= 0.3 * fatigue_damage

    # 8. Penalty for wasting mana (ending turn with significant unspent mana)
    if action[0] == "end":
        wasted_mana = prev_current_player.mana
        # Only penalize if player had playable cards or could have made attacks
        if wasted_mana > 1:  # Allow some flexibility for strategic mana saving
            reward -= 0.05 * wasted_mana

    # 9. Reward for efficient mana usage (playing cards)
    mana_spent = prev_current_player.mana - current_player.mana
    if mana_spent > 0 and action[0] == "play":
        reward += 0.02 * mana_spent

    # 10. Terminal rewards (most important)
    if game.is_terminal():
        if game.winner == acting_player_idx:
            reward += 2.0  # Increased win reward
        elif game.winner is not None:
            reward -= 2.0  # Increased loss penalty
        # Draw is neutral (reward = 0)

    return reward


def dqn_self_play():
    # Curriculum: start with only random agent, then add self-play after threshold
    curriculum_threshold = 0.7  # Win rate vs random to advance
    curriculum_phase = 0  # 0: only random, 1: mix random and self-play

    agent = DQNAgent(
        STATE_SIZE,
        ACTION_SPACE_SIZE,
        lr=LR,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        epsilon_decay=EPSILON_DECAY,
        prioritized_replay=True,  # Enable PER
    )

    target_agent = DQNAgent(
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

    win_history = []
    baseline_history = []
    best_win_rate = 0.0

    for episode in range(1, EPISODES + 1):
        # Curriculum logic
        if curriculum_phase == 0:
            opponent = RandomAgent()
        else:
            if random.random() < 0.3:  # Keep some random opponents for diversity
                opponent = RandomAgent()
            else:
                opponent = target_agent

        # Randomly assign who goes first
        agents = [agent, opponent]
        random.shuffle(agents)

        # Track which player index corresponds to our training agent
        agent_player_idx = 0 if agents[0] is agent else 1

        game = setup_game()
        game.players[0].start_turn()

        # Game loop with proper state tracking
        while not game.is_terminal():
            current_player_idx = game.current
            acting_agent = agents[current_player_idx]

            # Get current state
            state = encode_state(game, current_player_idx)

            # Get legal actions and create action mapping
            legal_actions = game.get_legal_actions()
            action_map, _ = get_action_index_map(game, legal_actions)

            # Choose action based on the acting agent
            if acting_agent is agent:
                action_idx = agent.select_action(state, list(action_map.keys()))
            else:
                # Handle different opponent types
                if hasattr(acting_agent, "select_action"):
                    try:
                        with torch.no_grad():
                            action_idx = acting_agent.select_action(
                                state, list(action_map.keys())
                            )
                    except:
                        action_idx = random.choice(list(action_map.keys()))
                else:
                    # Random agent or other simple agents
                    action_idx = random.choice(list(action_map.keys()))

            action = action_map[action_idx]

            # Store previous game state for reward calculation
            import copy

            prev_game_state = copy.deepcopy(game)

            # Execute action
            game.step(action)

            # Get next state
            next_state = encode_state(game, game.current)

            # Calculate reward only for our training agent
            if acting_agent is agent:
                reward = calculate_reward(
                    game, prev_game_state, action, current_player_idx
                )
                done = game.is_terminal()

                # Store transition
                agent.store_transition(state, action_idx, reward, next_state, done)

                # Update agent
                agent.update()

        if episode % 100 == 0 or episode == 1:
            print(f"Episode {episode}/{EPISODES} finished.")

        # Update target network
        if episode % TARGET_UPDATE == 0:
            agent.update_target()
            target_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Target network updated at episode {episode}.")

        # Save checkpoints
        if episode % 1000 == 0:
            torch.save(
                agent.policy_net.state_dict(), f"dqn_policy/dqn_policy_ep{episode}.pt"
            )
            print(f"Checkpoint saved at episode {episode}.")

        # Evaluation
        if episode % EVAL_INTERVAL == 0:
            # Self-play evaluation
            wins = 0
            for _ in range(100):
                eval_agents = [agent, target_agent]
                random.shuffle(eval_agents)
                agent_idx = 0 if eval_agents[0] is agent else 1

                g = setup_game()
                g.players[0].start_turn()

                while not g.is_terminal():
                    current_idx = g.current
                    acting_agent = eval_agents[current_idx]

                    legal = g.get_legal_actions()
                    amap, _ = get_action_index_map(g, legal)
                    s = encode_state(g, current_idx)

                    if acting_agent is agent:
                        # Use greedy policy for evaluation
                        old_epsilon = agent.epsilon
                        agent.epsilon = 0.0
                        aidx = agent.select_action(s, list(amap.keys()))
                        agent.epsilon = old_epsilon
                    else:
                        with torch.no_grad():
                            target_agent.epsilon = 0.0
                            aidx = target_agent.select_action(s, list(amap.keys()))

                    a = amap[aidx]
                    g.step(a)

                if g.winner == agent_idx:
                    wins += 1

            win_rate = wins / 100
            win_history.append(win_rate)
            print(f"Episode {episode}: DQN win rate vs self: {win_rate:.2f}")

            # Baseline evaluation vs RandomAgent
            baseline_wins = 0
            for _ in range(100):
                r = RandomAgent()
                eval_agents = [agent, r]
                random.shuffle(eval_agents)
                agent_idx = 0 if eval_agents[0] is agent else 1

                g = setup_game()
                g.players[0].start_turn()

                while not g.is_terminal():
                    current_idx = g.current
                    acting_agent = eval_agents[current_idx]

                    legal = g.get_legal_actions()
                    amap, _ = get_action_index_map(g, legal)
                    s = encode_state(g, current_idx)

                    if acting_agent is agent:
                        # Use greedy policy for evaluation
                        old_epsilon = agent.epsilon
                        agent.epsilon = 0.0
                        aidx = agent.select_action(s, list(amap.keys()))
                        agent.epsilon = old_epsilon
                    else:
                        aidx = random.choice(list(amap.keys()))

                    a = amap[aidx]
                    g.step(a)

                if g.winner == agent_idx:
                    baseline_wins += 1

            baseline_win_rate = baseline_wins / 100
            baseline_history.append(baseline_win_rate)
            print(f"Episode {episode}: DQN win rate vs random: {baseline_win_rate:.2f}")

            # Curriculum phase transition
            if curriculum_phase == 0 and baseline_win_rate > curriculum_threshold:
                print(f"Curriculum advanced to phase 1 at episode {episode}!")
                curriculum_phase = 1

            # Save best model
            if baseline_win_rate > best_win_rate:
                best_win_rate = baseline_win_rate
                torch.save(
                    agent.policy_net.state_dict(),
                    f"dqn_policy/dqn_policy_best_ep{episode}.pt",
                )
                print(
                    f"New best model saved at episode {episode} with win rate {baseline_win_rate:.2f}"
                )

            # Early stopping check
            if len(baseline_history) > 10 and all(
                baseline_history[-i] > 0.8 for i in range(1, 6)
            ):
                print("Early stopping: agent performing well consistently.")
                break

    # Plotting
    print("Training complete.")
    if win_history:
        x = list(
            range(EVAL_INTERVAL, EVAL_INTERVAL * len(win_history) + 1, EVAL_INTERVAL)
        )
        plt.figure(figsize=(12, 6))
        plt.plot(x, win_history, label="Self-play win rate", linewidth=2)
        plt.plot(x, baseline_history, label="Random baseline win rate", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Win Rate")
        plt.title("DQN Win Rate Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("dqn_policy/win_rate_curve.png", dpi=300, bbox_inches="tight")
        plt.show()

    # Save final model
    torch.save(agent.policy_net.state_dict(), "dqn_policy/dqn_policy_final.pt")
    print(f"Final model saved with best win rate: {best_win_rate:.2f}")


if __name__ == "__main__":
    # Create directory if it doesn't exist
    import os

    os.makedirs("dqn_policy", exist_ok=True)

    dqn_self_play()
