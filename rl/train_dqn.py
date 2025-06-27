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


def dqn_self_play():
    # Curriculum: start with only random agent, then add self-play after threshold
    curriculum_threshold = 0.7  # Win rate vs random to advance
    curriculum_phase = 0  # 0: only random, 1: mix random and self-play
    agent = DQNAgent(
        STATE_SIZE,
        ACTION_SPACE_SIZE,
        lr=5e-5,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        epsilon_decay=0.9995,
        prioritized_replay=True,  # Enable PER
    )
    target_agent = DQNAgent(
        STATE_SIZE,
        ACTION_SPACE_SIZE,
        lr=5e-5,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        epsilon_decay=0.9995,
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
            if random.random() < 0.2:
                opponent = RandomAgent()
            else:
                opponent = target_agent
        agents = [agent, opponent]
        random.shuffle(agents)

        game = setup_game()
        game.players[0].start_turn()
        state = encode_state(game, 0)
        done = False
        player_idx = 0

        while not game.is_terminal():
            legal_actions = game.get_legal_actions()
            action_map, _ = get_action_index_map(game, legal_actions)
            acting_agent = agents[game.current]
            if acting_agent is agent:
                action_idx = agent.select_action(state, list(action_map.keys()))
            else:
                try:
                    with torch.no_grad():
                        action_idx = acting_agent.select_action(
                            state, list(action_map.keys())
                        )
                except Exception:
                    action_idx = random.choice(list(action_map.keys()))
            action = action_map[action_idx]
            # --- Reward shaping logic ---
            prev_hero_hp = [p.hero_hp for p in game.players]
            prev_hand_sizes = [len(p.hand) for p in game.players]
            prev_board_sizes = [len(p.board) for p in game.players]
            prev_mana = [p.mana for p in game.players]
            prev_opp_board_objs = [m for m in game.players[1 - game.current].board]
            prev_own_board_objs = [m for m in game.players[game.current].board]
            game.step(action)
            next_state = encode_state(game, game.current)
            done = game.is_terminal()
            curr_hero_hp = [p.hero_hp for p in game.players]
            curr_hand_sizes = [len(p.hand) for p in game.players]
            curr_board_sizes = [len(p.board) for p in game.players]
            curr_mana = [p.mana for p in game.players]
            opp = 1 - player_idx
            reward = 0
            # Reward for dealing damage to opponent hero
            damage_dealt = prev_hero_hp[opp] - curr_hero_hp[opp]
            if damage_dealt > 0:
                reward += 0.1 * damage_dealt
            # Reward for playing a minion
            if curr_board_sizes[player_idx] > prev_board_sizes[player_idx]:
                reward += 0.05
            # Penalty for overdrawing (hand size decreases and hero HP decreases)
            if (
                curr_hand_sizes[player_idx] < prev_hand_sizes[player_idx]
                and curr_hero_hp[player_idx] < prev_hero_hp[player_idx]
            ):
                reward -= 0.05
            # Penalty for wasting mana (ending turn with unspent mana)
            if action[0] == "end" and prev_mana[player_idx] > 0:
                reward -= 0.05 * prev_mana[player_idx]
            # Reward for killing opponent minions (weighted by stats)
            killed_opp_minions = [
                m for m in prev_opp_board_objs if m not in game.players[opp].board
            ]
            for m in killed_opp_minions:
                reward += 0.1 * (m.attack + m.health)
            # Penalty for losing own minions
            killed_own_minions = [
                m
                for m in prev_own_board_objs
                if m not in game.players[player_idx].board
            ]
            for m in killed_own_minions:
                reward -= 0.1 * (m.attack + m.health)
            # Terminal rewards
            if done:
                if game.winner == player_idx:
                    reward += 1.0
                elif game.winner is not None:
                    reward -= 1.0
            if acting_agent is agent:
                agent.store_transition(state, action_idx, reward, next_state, done)
                agent.update()
            state = next_state
            player_idx = game.current

        if episode % 100 == 0 or episode == 1:
            print(f"Episode {episode}/{EPISODES} finished.")

        if episode % TARGET_UPDATE == 0:
            agent.update_target()
            target_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Target network updated at episode {episode}.")

        if episode % 1000 == 0:
            torch.save(
                agent.policy_net.state_dict(), f"dqn_policy/dqn_policy_ep{episode}.pt"
            )
            print(f"Checkpoint saved at episode {episode}.")

        if episode % EVAL_INTERVAL == 0:
            # Self-play evaluation
            wins = 0
            for _ in range(100):
                agents = [agent, target_agent]
                random.shuffle(agents)
                g = setup_game()
                g.players[0].start_turn()
                s = encode_state(g, 0)
                while not g.is_terminal():
                    legal = g.get_legal_actions()
                    amap, _ = get_action_index_map(g, legal)
                    acting_agent = agents[g.current]
                    if acting_agent is agent:
                        aidx = agent.select_action(s, list(amap.keys()))
                    else:
                        with torch.no_grad():
                            aidx = target_agent.select_action(s, list(amap.keys()))
                    a = amap[aidx]
                    g.step(a)
                    s = encode_state(g, g.current)
                if g.winner is not None and agents[g.winner] is agent:
                    wins += 1
            win_rate = wins / 100
            win_history.append(win_rate)
            print(f"Episode {episode}: DQN win rate vs self: {win_rate:.2f}")

            # Baseline evaluation vs RandomAgent
            baseline_wins = 0
            for _ in range(100):
                r = RandomAgent()
                agents = [agent, r]
                random.shuffle(agents)
                g = setup_game()
                g.players[0].start_turn()
                s = encode_state(g, 0)
                while not g.is_terminal():
                    legal = g.get_legal_actions()
                    amap, _ = get_action_index_map(g, legal)
                    acting_agent = agents[g.current]
                    if acting_agent is agent:
                        aidx = agent.select_action(s, list(amap.keys()))
                    else:
                        aidx = random.choice(list(amap.keys()))
                    a = amap[aidx]
                    g.step(a)
                    s = encode_state(g, g.current)
                if g.winner is not None and agents[g.winner] is agent:
                    baseline_wins += 1
            baseline_win_rate = baseline_wins / 100
            baseline_history.append(baseline_win_rate)
            print(f"Episode {episode}: DQN win rate vs random: {baseline_win_rate:.2f}")

            # Curriculum phase transition
            if curriculum_phase == 0 and baseline_win_rate > curriculum_threshold:
                print(f"Curriculum advanced to phase 1 at episode {episode}!")
                curriculum_phase = 1

            if baseline_win_rate > best_win_rate:
                best_win_rate = baseline_win_rate
                torch.save(
                    agent.policy_net.state_dict(),
                    f"dqn_policy/dqn_policy_best_ep{episode}.pt",
                )
                print(
                    f"New best model saved at episode {episode} with win rate {baseline_win_rate:.2f}"
                )

            if len(win_history) > 5 and all(
                abs(win_history[-i] - win_history[-i - 1]) < 0.01 for i in range(1, 6)
            ):
                print("Early stopping: win rate plateaued.")
                break

    # Plotting
    print("Training complete.")
    x = list(range(EVAL_INTERVAL, EVAL_INTERVAL * len(win_history) + 1, EVAL_INTERVAL))
    plt.figure(figsize=(10, 5))
    plt.plot(x, win_history, label="Self-play win rate")
    plt.plot(x, baseline_history, label="Random baseline win rate")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("DQN Win Rate Over Time")
    plt.legend()
    plt.grid()
    plt.savefig("dqn_policy/win_rate_curve.png")
    plt.show()
    torch.save(agent.policy_net.state_dict(), "dqn_policy/dqn_policy.pt")


def train_dqn_with_params(lr, batch_size, buffer_size, epsilon_decay):
    agent = DQNAgent(
        STATE_SIZE,
        ACTION_SPACE_SIZE,
        lr=lr,
        gamma=GAMMA,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epsilon_decay=epsilon_decay,
    )
    target_agent = DQNAgent(
        STATE_SIZE,
        ACTION_SPACE_SIZE,
        lr=lr,
        gamma=GAMMA,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epsilon_decay=epsilon_decay,
    )
    target_agent.policy_net.load_state_dict(agent.policy_net.state_dict())
    best_baseline_win_rate = 0.0
    for episode in range(1, 3000 + 1):  # Shorter for tuning
        agents = [agent, target_agent]
        random.shuffle(agents)
        game = setup_game()
        game.players[0].start_turn()
        state = encode_state(game, 0)
        done = False
        player_idx = 0
        prev_hero_hp = [p.hero_hp for p in game.players]
        prev_hand_sizes = [len(p.hand) for p in game.players]
        prev_board_sizes = [len(p.board) for p in game.players]
        prev_mana = [p.mana for p in game.players]
        prev_opp_board_objs = [m for m in game.players[1].board]
        prev_own_board_objs = [m for m in game.players[0].board]
        while not game.is_terminal():
            legal_actions = game.get_legal_actions()
            action_map, reverse_map = get_action_index_map(game, legal_actions)
            acting_agent = agents[game.current]
            if acting_agent is agent:
                action_idx = agent.select_action(state, list(action_map.keys()))
            else:
                with torch.no_grad():
                    action_idx = target_agent.select_action(
                        state, list(action_map.keys())
                    )
            action = action_map[action_idx]
            game.step(action)
            next_state = encode_state(game, game.current)
            reward = 0
            done = game.is_terminal()
            curr_hero_hp = [p.hero_hp for p in game.players]
            curr_hand_sizes = [len(p.hand) for p in game.players]
            curr_board_sizes = [len(p.board) for p in game.players]
            curr_mana = [p.mana for p in game.players]
            opp = 1 - player_idx
            damage_dealt = prev_hero_hp[opp] - curr_hero_hp[opp]
            if damage_dealt > 0:
                reward += 0.1 * damage_dealt
            if curr_board_sizes[player_idx] > prev_board_sizes[player_idx]:
                reward += 0.05
            if (
                curr_hand_sizes[player_idx] < prev_hand_sizes[player_idx]
                and curr_hero_hp[player_idx] < prev_hero_hp[player_idx]
            ):
                reward -= 0.05
            if action[0] == "end" and prev_mana[player_idx] > 0:
                reward -= 0.05 * prev_mana[player_idx]
            killed_opp_minions = [
                m for m in prev_opp_board_objs if m not in game.players[opp].board
            ]
            for m in killed_opp_minions:
                reward += 0.1 * (m.attack + m.health)
            killed_own_minions = [
                m
                for m in prev_own_board_objs
                if m not in game.players[player_idx].board
            ]
            for m in killed_own_minions:
                reward -= 0.1 * (m.attack + m.health)
            if done:
                if game.winner == player_idx:
                    reward += 1.0
                elif game.winner is not None:
                    reward -= 1.0
            if acting_agent is agent:
                agent.store_transition(state, action_idx, reward, next_state, done)
                agent.update()
            state = next_state
            player_idx = game.current
            prev_hero_hp = curr_hero_hp
            prev_hand_sizes = curr_hand_sizes
            prev_board_sizes = curr_board_sizes
            prev_mana = curr_mana
        # Evaluate vs random agent every 1000 episodes
        if episode % 1000 == 0:
            baseline_wins = 0
            for _ in range(50):
                from hs_core.random_agent import RandomAgent

                random_agent = RandomAgent()
                agents = [agent, random_agent]
                random.shuffle(agents)
                g = setup_game()
                g.players[0].start_turn()
                s = encode_state(g, 0)
                while not g.is_terminal():
                    legal = g.get_legal_actions()
                    amap, _ = get_action_index_map(g, legal)
                    acting_agent = agents[g.current]
                    if acting_agent is agent:
                        aidx = agent.select_action(s, list(amap.keys()))
                    else:
                        aidx = random.choice(list(amap.keys()))
                    a = amap[aidx]
                    g.step(a)
                    s = encode_state(g, g.current)
                if g.winner == 0 and agents[0] is agent:
                    baseline_wins += 1
                elif g.winner == 1 and agents[1] is agent:
                    baseline_wins += 1
            baseline_win_rate = baseline_wins / 50
            if baseline_win_rate > best_baseline_win_rate:
                best_baseline_win_rate = baseline_win_rate
    return best_baseline_win_rate


def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000])
    epsilon_decay = trial.suggest_uniform("epsilon_decay", 0.995, 0.99999)
    return train_dqn_with_params(lr, batch_size, buffer_size, epsilon_decay)


def run_optuna_tuning():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best trial:", study.best_trial)
    # Visualize optimization history and parameter importance
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()
    except Exception as e:
        print("Visualization failed:", e)


if __name__ == "__main__":
    dqn_self_play()
    # To run Optuna tuning, call run_optuna_tuning() instead
