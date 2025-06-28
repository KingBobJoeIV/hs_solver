import torch
import torch.nn as nn
import numpy as np
from rl.improved_dqn_agent import ImprovedDQNAgent
from rl.state_encoder import encode_state
from toy_hearthstone import setup_game, describe_action, print_board
from hs_core.random_agent import RandomAgent
from hs_core.scripted_agents import (
    AggroScriptedAgent,
    BoardControlScriptedAgent,
    BalancedScriptedAgent,
)
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau  # dynamic LR scheduling
import matplotlib.pyplot as plt
import copy
import os
import math
import time
from collections import deque
import rl.constants as constants
from rl.utils import get_beta, get_learning_rate, enhanced_encode_state


def enhanced_calculate_reward(game, prev_game_state, action, acting_player_idx):
    """Enhanced reward function with better strategic understanding and coin synergies"""
    reward = 0.0

    current_player = game.players[acting_player_idx]
    opponent = game.players[1 - acting_player_idx]
    prev_current_player = prev_game_state.players[acting_player_idx]
    prev_opponent = prev_game_state.players[1 - acting_player_idx]

    # Hero damage (primary objective)
    hero_damage = prev_opponent.hero_hp - opponent.hero_hp
    if hero_damage > 0:
        reward += constants.REWARD_CONFIG["hero_damage"] * hero_damage

    # Self damage penalty
    self_damage = prev_current_player.hero_hp - current_player.hero_hp
    if self_damage > 0:
        reward += constants.REWARD_CONFIG["self_damage"] * self_damage

    # Board control
    board_increase = len(current_player.board) - len(prev_current_player.board)
    if board_increase > 0:
        reward += constants.REWARD_CONFIG["board_presence"] * board_increase

    # Minion trading evaluation
    prev_opp_minions = {id(m): m for m in prev_opponent.board}
    curr_opp_minions = {id(m): m for m in opponent.board}

    for minion_id, minion in prev_opp_minions.items():
        if minion_id not in curr_opp_minions:
            reward += (
                constants.REWARD_CONFIG["kill_minion"]
                * (minion.attack + minion.health)
                / 6.0
            )

    prev_own_minions = {id(m): m for m in prev_current_player.board}
    curr_own_minions = {id(m): m for m in current_player.board}

    for minion_id, minion in prev_own_minions.items():
        if minion_id not in curr_own_minions:
            reward += (
                constants.REWARD_CONFIG["lose_minion"]
                * (minion.attack + minion.health)
                / 6.0
            )

    # Resource management penalties
    deck_decrease = len(prev_current_player.deck) - len(current_player.deck)
    hand_increase = len(current_player.hand) - len(prev_current_player.hand)

    if deck_decrease > 0 and hand_increase == 0 and len(prev_current_player.hand) >= 4:
        reward += constants.REWARD_CONFIG["overdraw"]

    fatigue_damage = current_player.fatigue - prev_current_player.fatigue
    if fatigue_damage > 0:
        reward += constants.REWARD_CONFIG["fatigue"] * fatigue_damage

    # Mana efficiency
    if action[0] == "end":
        wasted_mana = prev_current_player.mana
        # Penalize any unspent mana if there are playable cards
        playable = False
        for idx, card in enumerate(prev_current_player.hand):
            if hasattr(card, "mana") and card.mana <= prev_current_player.mana:
                playable = True
                break
        if wasted_mana > 0 and playable:
            reward += constants.REWARD_CONFIG["mana_waste"] * wasted_mana
        # Optionally, small bonus for using all mana
        if wasted_mana == 0:
            reward += 0.1  # Encourage using all mana

    mana_spent = prev_current_player.mana - current_player.mana
    if mana_spent > 0 and action[0] == "play":
        reward += constants.REWARD_CONFIG["mana_efficiency"] * mana_spent

    # Add coin usage reward for second player
    coin_used_this_turn = False
    coin_killed_minion = False
    coin_board_swing = False
    if action[0] == "play":
        card = (
            prev_current_player.hand[action[1]]
            if len(prev_current_player.hand) > action[1]
            else None
        )
        if (
            card is not None
            and getattr(card, "name", None) == "The Coin"
            and not getattr(prev_current_player, "is_first", True)
        ):
            reward += constants.REWARD_CONFIG.get("coin_usage", 0.0)
            coin_used_this_turn = True
        # Bonus for using the coin to play an oversized minion
        if (
            card is not None
            and getattr(card, "card_type", None) == "minion"
            and not prev_current_player.is_first
        ):
            if card.mana > prev_current_player.max_mana:
                reward += constants.REWARD_CONFIG.get("coin_play_minion", 0.0)

    # --- Coin synergy rewards ---
    # If Coin was used this turn, check for board swing or removal
    if not prev_current_player.is_first:
        # Detect if Coin was used this turn (mana increased by 1, Coin left hand)
        prev_has_coin = any(
            getattr(card, "name", None) == "The Coin"
            for card in prev_current_player.hand
        )
        curr_has_coin = any(
            getattr(card, "name", None) == "The Coin" for card in current_player.hand
        )
        coin_used_this_turn = prev_has_coin and not curr_has_coin
        # Board swing: did our board size increase and/or opponent's decrease?
        board_delta = (len(current_player.board) - len(prev_current_player.board)) - (
            len(opponent.board) - len(prev_opponent.board)
        )
        if coin_used_this_turn and board_delta > 0:
            reward += constants.REWARD_CONFIG.get("coin_board_swing", 0.0)
            coin_board_swing = True
        # Coin+removal: did we kill an enemy minion this turn and use Coin?
        prev_opp_minions = {id(m): m for m in prev_opponent.board}
        curr_opp_minions = {id(m): m for m in opponent.board}
        killed_minions = [
            minion
            for minion_id, minion in prev_opp_minions.items()
            if minion_id not in curr_opp_minions
        ]
        if coin_used_this_turn and killed_minions:
            reward += constants.REWARD_CONFIG.get("coin_removal", 0.0)
            coin_killed_minion = True

    # Terminal rewards (most important)
    if game.is_terminal():
        if game.winner == acting_player_idx:
            reward += constants.REWARD_CONFIG["win_reward"]
        elif game.winner is not None:
            reward += constants.REWARD_CONFIG["loss_penalty"]

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
    print(f"Action space size: {constants.ACTION_SPACE_SIZE}")
    print(f"Training for {constants.EPISODES} episodes")

    # Initialize agents
    agent = ImprovedDQNAgent(
        STATE_SIZE,
        constants.ACTION_SPACE_SIZE,
        lr=constants.LR,
        gamma=constants.GAMMA,
        buffer_size=constants.BUFFER_SIZE,
        batch_size=constants.BATCH_SIZE,
        epsilon_start=constants.EPSILON_START,
        epsilon_end=constants.EPSILON_END,
        epsilon_decay=constants.EPSILON_DECAY,
        prioritized_replay=True,
        alpha=constants.ALPHA,
        beta=constants.BETA_START,
    )

    target_agent = ImprovedDQNAgent(
        STATE_SIZE,
        constants.ACTION_SPACE_SIZE,
        lr=constants.LR,
        gamma=constants.GAMMA,
        buffer_size=constants.BUFFER_SIZE,
        batch_size=constants.BATCH_SIZE,
        epsilon_decay=constants.EPSILON_DECAY,
        prioritized_replay=True,
    )
    target_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
    # Set up learning-rate scheduler and initial checkpoint
    scheduler = ReduceLROnPlateau(
        agent.optimizer,
        mode="max",  # maximize win rate
        factor=0.5,  # drop LR by half on plateau
        patience=2,  # wait 2 eval intervals
        verbose=True,
    )
    best_checkpoint_rate = 0.0
    torch.save(agent.policy_net.state_dict(), "dqn_policy/best.pt")
    # Set up LR scheduler on plateau and initial checkpoint
    scheduler = ReduceLROnPlateau(
        agent.optimizer,
        mode="max",  # maximize win rate
        factor=0.5,  # reduce LR by half on plateau
        patience=2,  # wait 2 eval intervals
        verbose=True,
    )
    # Track best checkpoint for rollback
    best_checkpoint_rate = 0.0
    torch.save(agent.policy_net.state_dict(), "dqn_policy/best.pt")

    # Opponent pool for advanced self-play
    opponent_pool = [copy.deepcopy(target_agent)]

    # Scripted agents
    aggro_agent = AggroScriptedAgent()
    board_control_agent = BoardControlScriptedAgent()
    balanced_agent = BalancedScriptedAgent()
    scripted_opponents = [aggro_agent, board_control_agent, balanced_agent]

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
    print(f"Epsilon decay rate: {constants.EPSILON_DECAY:.6f}")
    print("-" * 50)

    for episode in range(1, constants.EPISODES + 1):
        # Periodic epsilon reset to encourage exploration
        if episode % 2000 == 0:
            agent.epsilon = max(agent.epsilon, 0.15)
        # Curriculum opponent selection with advanced self-play and scripted agents
        if curriculum_phase == 0:
            # 80% random, 20% scripted
            if random.random() < 0.2:
                opponent = random.choice(scripted_opponents)
            else:
                opponent = RandomAgent()
        else:
            # 5% random, 10% scripted, 85% self-play from pool
            r = random.random()
            if r < 0.05:
                opponent = RandomAgent()
            elif r < 0.15:
                opponent = random.choice(scripted_opponents)
            else:
                opponent = random.choice(opponent_pool)

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
            if acting_agent is agent and episode > constants.WARMUP_EPISODES:
                next_state = enhanced_encode_state(game, game.current)
                reward = enhanced_calculate_reward(
                    game, prev_game_state, action, current_player_idx
                )

                # Add turn length penalty for very long games
                if episode_length > 50:
                    reward += constants.REWARD_CONFIG["turn_length_penalty"]

                done = game.is_terminal()
                episode_reward += reward

                # Update beta for prioritized replay
                agent.beta = get_beta(episode)

                agent.store_transition(state, action_idx, reward, next_state, done)
                agent.update()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Update learning rate
        if episode > constants.WARMUP_EPISODES:
            agent.update_learning_rate(episode)

        # Target network update
        if episode % constants.TARGET_UPDATE == 0:
            agent.update_target()
            target_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
            # Add snapshot to opponent pool for self-play
            if curriculum_phase > 0:
                opponent_pool.append(copy.deepcopy(target_agent))
                # Limit pool size to last 5
                if len(opponent_pool) > 5:
                    opponent_pool.pop(0)

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
        if (
            episode % constants.EVAL_INTERVAL == 0
            and episode > constants.WARMUP_EPISODES
        ):
            # Evaluate against random agent
            baseline_win_rate = evaluate_agent(agent, "random", num_games=200)
            baseline_history.append(baseline_win_rate)

            # Evaluate against scripted agents
            scripted_winrates = []
            for scripted in scripted_opponents:
                wr = evaluate_agent(agent, scripted, num_games=100)
                scripted_winrates.append(wr)

            # Self-play evaluation (vs current target agent)
            if curriculum_phase > 0:
                self_play_win_rate = evaluate_agent(agent, target_agent, num_games=100)
                win_history.append(self_play_win_rate)
            else:
                self_play_win_rate = 0.5  # Placeholder during phase 0
                win_history.append(0.5)

            # Evaluation vs pool of past agents (excluding latest target_agent)
            if curriculum_phase > 0 and len(opponent_pool) > 1:
                pool_winrates = []
                for idx, past_agent in enumerate(opponent_pool[:-1]):  # Exclude latest
                    winrate = evaluate_agent(agent, past_agent, num_games=50)
                    pool_winrates.append(winrate)
                avg_pool_winrate = np.mean(pool_winrates) if pool_winrates else 0
            else:
                pool_winrates = []
                avg_pool_winrate = None

            epsilon_history.append(agent.epsilon)

            print(f"\nEvaluation at episode {episode}:")
            print(f"  Win rate vs Random: {baseline_win_rate:.3f}")
            print(f"  Win rate vs AggroScriptedAgent: {scripted_winrates[0]:.3f}")
            print(
                f"  Win rate vs BoardControlScriptedAgent: {scripted_winrates[1]:.3f}"
            )
            print(f"  Win rate vs BalancedScriptedAgent: {scripted_winrates[2]:.3f}")
            if curriculum_phase > 0:
                print(f"  Win rate vs Self:   {self_play_win_rate:.3f}")
                if avg_pool_winrate is not None:
                    print(f"  Win rate vs pool of past agents: {avg_pool_winrate:.3f}")
                    for i, wr in enumerate(pool_winrates):
                        print(f"    vs pool agent {i}: {wr:.3f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Learning rate: {get_learning_rate(episode):.6f}")
            print("-" * 50)

            # Compute composite evaluation score across opponents
            metrics = [baseline_win_rate] + scripted_winrates
            if curriculum_phase > 0:
                metrics.append(self_play_win_rate)
                metrics.extend(pool_winrates)
            overall_score = np.mean(metrics)
            # Scheduler on composite score
            scheduler.step(overall_score)
            # Checkpoint & rollback based on composite score
            if overall_score > best_checkpoint_rate:
                best_checkpoint_rate = overall_score
                torch.save(agent.policy_net.state_dict(), "dqn_policy/best.pt")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= constants.EARLY_STOP_PATIENCE:
                    print("↩️ Rollback to best policy due to plateau")
                    agent.policy_net.load_state_dict(
                        torch.load("dqn_policy/best.pt", map_location=agent.device)
                    )
                    agent.update_target()
                    agent.epsilon = min(agent.epsilon + 0.2, constants.EPSILON_START)
                    early_stop_counter = 0

            # Curriculum advancement
            if curriculum_phase == 0 and overall_score > constants.CURRICULUM_THRESHOLD:
                print(f"🎓 Advancing to curriculum phase 1 at episode {episode}!")
                curriculum_phase = 1

            # Save best model (composite score)
            if overall_score > best_win_rate:
                improvement = overall_score - best_win_rate
                best_win_rate = overall_score
                torch.save(
                    agent.policy_net.state_dict(),
                    f"dqn_policy/dqn_policy_best_ep{episode}.pt",
                )
                print(
                    f"💾 New best model saved! Composite score: {overall_score:.3f} (+{improvement:.3f})"
                )
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Early stopping (require minimum episodes)
            if (
                overall_score >= constants.TARGET_WIN_RATE
                and episode >= constants.MIN_TRAIN_EPISODES
            ):
                print(
                    f"🎯 Target composite score {constants.TARGET_WIN_RATE:.2f} achieved and minimum episodes reached! Stopping early."
                )
                break

            if (
                early_stop_counter >= constants.EARLY_STOP_PATIENCE
                and episode >= constants.MIN_TRAIN_EPISODES
            ):
                print(
                    f"⏹️  Early stopping: No improvement for {constants.EARLY_STOP_PATIENCE} evaluations and minimum episodes reached"
                )
                break

        # Save checkpoints
        if episode % constants.SAVE_INTERVAL == 0:
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
                constants.EVAL_INTERVAL,
                constants.EVAL_INTERVAL * len(baseline_history) + 1,
                constants.EVAL_INTERVAL,
            )
        )

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(episodes_eval, baseline_history, "b-", linewidth=2, label="vs Random")
        if len(win_history) > 1:
            plt.plot(episodes_eval, win_history, "r--", linewidth=2, label="vs Self")
        plt.axhline(
            y=constants.TARGET_WIN_RATE,
            color="g",
            linestyle=":",
            label=f"Target ({constants.TARGET_WIN_RATE})",
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
