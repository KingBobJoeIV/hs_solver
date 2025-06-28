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


def get_adaptive_epsilon(episode, total_episodes):
    """Multi-phase epsilon decay with periodic resets"""
    # Check for periodic reset
    if episode % constants.EPSILON_RESET_INTERVAL == 0 and episode > 0:
        return constants.EPSILON_RESET_VALUE

    progress = episode / total_episodes

    if progress < constants.EPSILON_PHASE_1_END:
        # Phase 1: Slow decay for thorough exploration
        phase_progress = progress / constants.EPSILON_PHASE_1_END
        return constants.EPSILON_START - 0.5 * phase_progress
    elif progress < constants.EPSILON_PHASE_2_END:
        # Phase 2: Moderate decay
        phase_progress = (progress - constants.EPSILON_PHASE_1_END) / (
            constants.EPSILON_PHASE_2_END - constants.EPSILON_PHASE_1_END
        )
        return 0.5 - 0.4 * phase_progress
    else:
        # Phase 3: Fine-tuning with minimal exploration
        phase_progress = (progress - constants.EPSILON_PHASE_2_END) / (
            1.0 - constants.EPSILON_PHASE_2_END
        )
        return 0.1 * (1 - phase_progress * 0.8)  # Keep some exploration


# New reward function: only terminal rewards, no intermediate rewards
def terminal_only_reward(game, prev_game_state, action, acting_player_idx):
    """Reward only for win/loss at terminal state, zero otherwise."""
    if game.is_terminal():
        if game.winner == acting_player_idx:
            return constants.REWARD_CONFIG["win_reward"]
        elif game.winner is not None:
            return constants.REWARD_CONFIG["loss_penalty"]
    return 0.0


def enhanced_calculate_reward(game, prev_game_state, action, acting_player_idx):
    """Enhanced reward function with improved strategic understanding"""
    reward = 0.0

    current_player = game.players[acting_player_idx]
    opponent = game.players[1 - acting_player_idx]
    prev_current_player = prev_game_state.players[acting_player_idx]
    prev_opponent = prev_game_state.players[1 - acting_player_idx]

    # Hero damage (primary objective) - significantly increased weight
    hero_damage = prev_opponent.hero_hp - opponent.hero_hp
    if hero_damage > 0:
        reward += constants.REWARD_CONFIG["hero_damage"] * hero_damage

    # Self damage penalty - stronger
    self_damage = prev_current_player.hero_hp - current_player.hero_hp
    if self_damage > 0:
        reward += constants.REWARD_CONFIG["self_damage"] * self_damage

    # Board control with tempo consideration
    board_increase = len(current_player.board) - len(prev_current_player.board)
    board_decrease_opp = len(prev_opponent.board) - len(opponent.board)

    if board_increase > 0:
        reward += constants.REWARD_CONFIG["board_presence"] * board_increase

    # Tempo swing bonus (gaining board while opponent loses it)
    tempo_swing = board_increase + board_decrease_opp
    if tempo_swing > 1:
        reward += constants.REWARD_CONFIG["tempo_swing"] * (tempo_swing - 1)

    # Enhanced minion trading evaluation
    prev_opp_minions = {id(m): m for m in prev_opponent.board}
    curr_opp_minions = {id(m): m for m in opponent.board}

    total_opp_stats_killed = 0
    for minion_id, minion in prev_opp_minions.items():
        if minion_id not in curr_opp_minions:
            minion_value = (minion.attack + minion.health) / 6.0
            reward += constants.REWARD_CONFIG["kill_minion"] * minion_value
            total_opp_stats_killed += minion.attack + minion.health

    prev_own_minions = {id(m): m for m in prev_current_player.board}
    curr_own_minions = {id(m): m for m in current_player.board}

    total_own_stats_lost = 0
    for minion_id, minion in prev_own_minions.items():
        if minion_id not in curr_own_minions:
            minion_value = (minion.attack + minion.health) / 6.0
            reward += constants.REWARD_CONFIG["lose_minion"] * minion_value
            total_own_stats_lost += minion.attack + minion.health

    # Efficient trade bonus (killed more stats than lost)
    if total_opp_stats_killed > 0 and total_own_stats_lost > 0:
        if total_opp_stats_killed > total_own_stats_lost:
            trade_efficiency = (total_opp_stats_killed - total_own_stats_lost) / 6.0
            reward += constants.REWARD_CONFIG["efficient_trade"] * trade_efficiency

    # Resource management penalties
    deck_decrease = len(prev_current_player.deck) - len(current_player.deck)
    hand_increase = len(current_player.hand) - len(prev_current_player.hand)

    if deck_decrease > 0 and hand_increase == 0 and len(prev_current_player.hand) >= 4:
        reward += constants.REWARD_CONFIG["overdraw"]

    fatigue_damage = current_player.fatigue - prev_current_player.fatigue
    if fatigue_damage > 0:
        reward += constants.REWARD_CONFIG["fatigue"] * fatigue_damage

    # Improved mana efficiency
    mana_spent = prev_current_player.mana - current_player.mana

    if action[0] == "end":
        wasted_mana = prev_current_player.mana
        # Check if there were actually playable cards
        playable = False
        for idx, card in enumerate(prev_current_player.hand):
            if hasattr(card, "mana") and card.mana <= prev_current_player.mana:
                playable = True
                break

        if wasted_mana > 0 and playable:
            reward += constants.REWARD_CONFIG["mana_waste"] * wasted_mana
        elif wasted_mana == 0:  # Perfect mana usage
            reward += constants.REWARD_CONFIG["mana_efficiency"]

    # Optimal curve play bonus (playing on-curve or above-curve)
    if action[0] == "play" and mana_spent > 0:
        current_turn = prev_current_player.turn
        if mana_spent >= min(current_turn, prev_current_player.max_mana):
            reward += constants.REWARD_CONFIG["optimal_curve"]

        reward += constants.REWARD_CONFIG["mana_efficiency"] * (mana_spent / 10.0)

    # Enhanced coin usage rewards
    if not prev_current_player.is_first:
        # Detect if Coin was used this turn
        prev_has_coin = any(
            getattr(card, "name", None) == "The Coin"
            for card in prev_current_player.hand
        )
        curr_has_coin = any(
            getattr(card, "name", None) == "The Coin" for card in current_player.hand
        )
        coin_used_this_turn = prev_has_coin and not curr_has_coin

        if coin_used_this_turn:
            reward += constants.REWARD_CONFIG["coin_usage"]

            # Bonus for tempo swing with coin
            if tempo_swing > 0:
                reward += constants.REWARD_CONFIG["coin_board_swing"]

            # Bonus for removal with coin
            if total_opp_stats_killed > 0:
                reward += constants.REWARD_CONFIG["coin_removal"]

    # Attack and lethal bonuses
    if action[0] == "attack":
        reward += constants.REWARD_CONFIG["attack_bonus"]

        # Lethal bonus
        if opponent.hero_hp <= 0:
            reward += constants.REWARD_CONFIG["lethal_bonus"]

    # Penalty for not attacking when possible
    if action[0] == "end":
        available_attackers = [m for m in prev_current_player.board if m.can_attack]
        if (
            available_attackers and opponent.hero_hp <= 10
        ):  # Only penalize when close to lethal
            reward += constants.REWARD_CONFIG["skip_attack_penalty"] * len(
                available_attackers
            )

    # Terminal rewards with turn efficiency
    if game.is_terminal():
        total_turns = prev_current_player.turn + prev_opponent.turn
        if game.winner == acting_player_idx:
            reward += constants.REWARD_CONFIG["win_reward"]
            # Bonus for quick wins
            if total_turns < 10:
                reward += constants.REWARD_CONFIG["turn_efficiency"] * (
                    10 - total_turns
                )
        elif game.winner is not None:
            reward += constants.REWARD_CONFIG["loss_penalty"]

    return reward


def get_action_index_map(game, legal_actions):
    """Map legal actions to indices for DQN"""
    action_map = {i: a for i, a in enumerate(legal_actions)}
    return action_map, {str(a): i for i, a in enumerate(legal_actions)}


def evaluate_agent(agent, opponent_type="random", num_games=200, verbose=False):
    """Evaluate agent performance with improved stability"""
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
                            if hasattr(acting_agent, "epsilon"):
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


def select_curriculum_opponent(
    episode, curriculum_phase, win_rate_history, opponent_pool, scripted_opponents
):
    """Adaptive opponent selection based on performance"""

    if curriculum_phase == 0:  # Early learning phase
        if random.random() < 0.85:
            return RandomAgent()
        else:
            return random.choice(scripted_opponents)

    elif curriculum_phase == 1:  # Intermediate phase
        # Adaptive difficulty based on recent performance
        recent_performance = (
            np.mean(win_rate_history[-5:]) if len(win_rate_history) >= 5 else 0.5
        )

        if recent_performance > 0.85:  # Doing well, increase difficulty
            weights = [0.1, 0.2, 0.7]  # [random, scripted, self-play]
        elif recent_performance < 0.7:  # Struggling, easier opponents
            weights = [0.4, 0.4, 0.2]
        else:  # Balanced
            weights = [0.2, 0.3, 0.5]

        choice = np.random.choice(["random", "scripted", "self"], p=weights)

        if choice == "random":
            return RandomAgent()
        elif choice == "scripted":
            return random.choice(scripted_opponents)
        else:
            return random.choice(opponent_pool) if opponent_pool else RandomAgent()

    else:  # Advanced phase (phase 2+)
        # Mostly self-play with occasional scripted opponents
        if random.random() < 0.1:
            return random.choice(scripted_opponents)
        else:
            return random.choice(opponent_pool) if opponent_pool else RandomAgent()


def comprehensive_evaluation(agent, opponent_pool, scripted_opponents):
    """Multi-faceted evaluation with weighted scoring"""
    evaluations = {}

    # Against different opponent types
    evaluations["random"] = evaluate_agent(agent, RandomAgent(), 200)
    evaluations["aggro"] = evaluate_agent(agent, scripted_opponents[0], 100)
    evaluations["control"] = evaluate_agent(agent, scripted_opponents[1], 100)
    evaluations["balanced"] = evaluate_agent(agent, scripted_opponents[2], 100)

    # Self-play against different strength opponents
    if len(opponent_pool) > 1:
        for i, past_agent in enumerate(opponent_pool[-3:]):  # Last 3 versions
            evaluations[f"self_play_{i}"] = evaluate_agent(agent, past_agent, 50)

    # Compute weighted composite score (emphasize harder opponents)
    weights = {
        "random": 0.15,
        "aggro": 0.25,
        "control": 0.25,
        "balanced": 0.25,
        "self_play": 0.1,
    }

    composite_score = 0.0
    total_weight = 0.0

    for key, score in evaluations.items():
        if key.startswith("self_play"):
            weight = weights["self_play"]
        else:
            weight = weights.get(key, 0.1)
        composite_score += score * weight
        total_weight += weight

    composite_score = composite_score / total_weight if total_weight > 0 else 0.0

    return evaluations, composite_score


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
    print(f"Using adaptive epsilon schedule with periodic resets")

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
        patience=3,  # increased patience
        verbose=True,
    )
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
    composite_score_history = []
    best_win_rate = 0.0
    curriculum_phase = 0
    early_stop_counter = 0

    # Training statistics
    episode_rewards = deque(maxlen=1000)
    episode_lengths = deque(maxlen=1000)

    start_time = time.time()

    print("Starting improved training...")
    print(
        f"Curriculum thresholds: Phase 1 at {constants.CURRICULUM_THRESHOLD_PHASE_1:.2f}, Phase 2 at {constants.CURRICULUM_THRESHOLD_PHASE_2:.2f}"
    )
    print("-" * 50)

    for episode in range(1, constants.EPISODES + 1):
        # Adaptive epsilon schedule
        agent.epsilon = get_adaptive_epsilon(episode, constants.EPISODES)

        # Curriculum opponent selection
        opponent = select_curriculum_opponent(
            episode,
            curriculum_phase,
            composite_score_history,
            opponent_pool,
            scripted_opponents,
        )

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
            if acting_agent is agent and episode > constants.REPLAY_BUFFER_WARMUP:
                next_state = enhanced_encode_state(game, game.current)
                reward = terminal_only_reward(
                    game, prev_game_state, action, current_player_idx
                )

                done = game.is_terminal()
                episode_reward += reward

                # Update beta for prioritized replay
                agent.beta = get_beta(episode)

                agent.store_transition(state, action_idx, reward, next_state, done)

                # Only update if we have enough experience
                if len(agent.memory) > constants.BATCH_SIZE:
                    loss = agent.update()
                    if loss is not None:
                        loss_history.append(loss)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Learning rate decay
        if episode % constants.LR_DECAY_INTERVAL == 0:
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] *= constants.LR_DECAY_FACTOR

        # Target network update
        if episode % constants.TARGET_UPDATE == 0:
            if constants.USE_SOFT_UPDATE:
                # Soft update
                for target_param, local_param in zip(
                    target_agent.policy_net.parameters(), agent.policy_net.parameters()
                ):
                    target_param.data.copy_(
                        constants.TAU * local_param.data
                        + (1.0 - constants.TAU) * target_param.data
                    )
            else:
                # Hard update
                target_agent.policy_net.load_state_dict(agent.policy_net.state_dict())

            # Add snapshot to opponent pool for self-play
            if curriculum_phase > 0:
                opponent_pool.append(copy.deepcopy(target_agent))
                # Limit pool size to last 6
                if len(opponent_pool) > 6:
                    opponent_pool.pop(0)

        # Logging
        if episode % 100 == 0 or episode == 1:
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            avg_loss = np.mean(loss_history[-100:]) if loss_history else 0
            print(
                f"Episode {episode:5d} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Avg Reward: {avg_reward:6.2f} | "
                f"Avg Length: {avg_length:4.1f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Phase: {curriculum_phase}"
            )

        # Evaluation
        if (
            episode % constants.EVAL_INTERVAL == 0
            and episode > constants.WARMUP_EPISODES
        ):
            # Comprehensive evaluation
            evaluations, composite_score = comprehensive_evaluation(
                agent, opponent_pool, scripted_opponents
            )

            composite_score_history.append(composite_score)
            baseline_history.append(evaluations["random"])

            # For backward compatibility
            if curriculum_phase > 0 and len(opponent_pool) > 1:
                self_play_win_rate = evaluations.get("self_play_0", 0.5)
            else:
                self_play_win_rate = 0.5
            win_history.append(self_play_win_rate)

            epsilon_history.append(agent.epsilon)

            print(f"\nEvaluation at episode {episode}:")
            print(f"  Composite Score: {composite_score:.3f}")
            for key, score in evaluations.items():
                print(f"  {key}: {score:.3f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Learning rate: {agent.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)

            # Scheduler step on composite score
            scheduler.step(composite_score)

            # Checkpoint & rollback based on composite score
            if composite_score > best_checkpoint_rate:
                best_checkpoint_rate = composite_score
                torch.save(agent.policy_net.state_dict(), "dqn_policy/best.pt")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= constants.EARLY_STOP_PATIENCE:
                    print("↩️ Rollback to best policy due to plateau")
                    agent.policy_net.load_state_dict(
                        torch.load("dqn_policy/best.pt", map_location=agent.device)
                    )
                    # Soft target update after rollback
                    if constants.USE_SOFT_UPDATE:
                        for target_param, local_param in zip(
                            target_agent.policy_net.parameters(),
                            agent.policy_net.parameters(),
                        ):
                            target_param.data.copy_(local_param.data)
                    else:
                        target_agent.policy_net.load_state_dict(
                            agent.policy_net.state_dict()
                        )
                    # Boost exploration after rollback
                    agent.epsilon = min(agent.epsilon + 0.2, constants.EPSILON_START)
                    early_stop_counter = 0

            # Curriculum advancement based on composite score
            if (
                curriculum_phase == 0
                and composite_score > constants.CURRICULUM_THRESHOLD_PHASE_1
            ):
                print(f"🎓 Advancing to curriculum phase 1 at episode {episode}!")
                curriculum_phase = 1
            elif (
                curriculum_phase == 1
                and composite_score > constants.CURRICULUM_THRESHOLD_PHASE_2
            ):
                print(f"🎓 Advancing to curriculum phase 2 at episode {episode}!")
                curriculum_phase = 2

            # Save best model (composite score)
            if composite_score > best_win_rate:
                improvement = composite_score - best_win_rate
                best_win_rate = composite_score
                torch.save(
                    agent.policy_net.state_dict(),
                    f"dqn_policy/dqn_policy_best_ep{episode}.pt",
                )
                print(
                    f"💾 New best model saved! Composite score: {composite_score:.3f} (+{improvement:.3f})"
                )
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Early stopping (require minimum episodes)
            if (
                composite_score >= constants.TARGET_WIN_RATE
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
    print(f"Best composite score achieved: {best_win_rate:.3f}")

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
