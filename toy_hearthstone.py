import random
import copy
from hs_core.card import Card
from hs_core.minion import Minion
from hs_core.player import Player
from hs_core.gamestate import GameState
from hs_core.utils import parse_effect
from rl.dqn_agent import DQNAgent
from rl.state_encoder import encode_state
from hs_core.dqn_play_agent import DQNPlayAgent
import torch
from hs_core.random_agent import RandomAgent

ALL_CARDS = [
    Card("1/1", 1, "minion", 1, 1),
    Card("1 mana deal 1", 1, "spell", effect="deal_1"),
    Card("2/2", 2, "minion", 2, 2),
    Card("2 1/1 deal 1", 2, "minion", 1, 1, effect="deal_1"),
    Card("3/3", 3, "minion", 3, 3),
    Card("3 2/2 deal 1", 3, "minion", 2, 2, effect="deal_1"),
    Card("3 mana deal 2", 3, "spell", effect="deal_2"),
    Card("The Coin", 0, "spell", effect="coin"),
]
CARD_LIBRARY = {c.name: c for c in ALL_CARDS}


def build_deck():
    return [
        CARD_LIBRARY["1/1"],
        CARD_LIBRARY["1 mana deal 1"],
        CARD_LIBRARY["2/2"],
        CARD_LIBRARY["2 1/1 deal 1"],
        CARD_LIBRARY["3/3"],
        CARD_LIBRARY["3 2/2 deal 1"],
        CARD_LIBRARY["3 mana deal 2"],
    ]


def setup_game():
    deck1 = build_deck()
    deck2 = build_deck()
    p1 = Player("P1", deck1, is_first=True)
    p2 = Player("P2", deck2, is_first=False)
    for _ in range(1):
        p1.draw()
    for _ in range(2):
        p2.draw()
    p2.hand.append(CARD_LIBRARY["The Coin"])
    return GameState(p1, p2)


def print_board(state):
    p1, p2 = state.players[0], state.players[1]
    print("\n==============================")
    print(f"Turn {p1.turn + p2.turn} - Player {state.current + 1}'s turn")
    for idx, p in enumerate([p1, p2], 1):
        print(
            f"P{idx}: HP={p.hero_hp} Mana={p.mana}/{p.max_mana} Hand={len(p.hand)} Deck={len(p.deck)}"
        )
        print("  Hand:", [card.name for card in p.hand])
        print("  Board:", [f"{m.card.name}({m.attack}/{m.health})" for m in p.board])
    print("==============================\n")


def describe_action(state, action, prev_state, player_idx):
    if action[0] == "end":
        return f"Player {player_idx + 1} ends their turn."
    elif action[0] == "play":
        card = prev_state.players[player_idx].hand[action[1]]
        target = action[2]
        effect_type, amount = parse_effect(card.effect)
        if card.card_type == "minion":
            desc = f"Player {player_idx + 1} plays minion '{card.name}'"
            if effect_type == "deal" and amount is not None:
                if target == "hero":
                    desc += f" and deals {amount} to opponent's hero."
                elif target is not None:
                    desc += f" and deals {amount} to opponent's minion {target}."
            else:
                desc += "."
        elif card.card_type == "spell":
            if effect_type == "deal" and amount is not None:
                if target == "hero":
                    desc = f"Player {player_idx + 1} casts '{card.name}' and deals {amount} to opponent's hero."
                elif target is not None:
                    desc = f"Player {player_idx + 1} casts '{card.name}' and deals {amount} to opponent's minion {target}."
            elif card.effect == "coin":
                desc = f"Player {player_idx + 1} plays The Coin for +1 mana."
            else:
                desc = f"Player {player_idx + 1} plays spell '{card.name}'."
        else:
            desc = f"Player {player_idx + 1} plays '{card.name}'."
        return desc
    elif action[0] == "attack":
        minion = prev_state.players[player_idx].board[action[1]]
        target = action[2]
        if target == "hero":
            return f"Player {player_idx + 1}'s minion '{minion.card.name}' attacks opponent's hero for {minion.attack}."
        else:
            return f"Player {player_idx + 1}'s minion '{minion.card.name}' attacks opponent's minion {target}."
    return str(action)


def play_game(agent1, agent2, visualize=False):
    state = setup_game()
    state.players[0].start_turn()
    if visualize:
        print_board(state)
    while not state.is_terminal():
        agent = agent1 if state.current == 0 else agent2
        while True:
            actions = state.get_legal_actions()
            non_end_actions = [a for a in actions if a[0] != "end"]
            if not non_end_actions:
                action = ("end",)
            else:
                action = agent.choose_action(state)
            prev_player = state.current
            prev_state = copy.deepcopy(state)
            state.step(action)
            if visualize:
                print(describe_action(state, action, prev_state, prev_player))
                print_board(state)
            if action[0] == "end" or state.is_terminal():
                break
    return state.winner


STATE_SIZE = 60  # Update if your encoder changes
ACTION_SPACE_SIZE = 50

if __name__ == "__main__":
    dqn_agent = DQNPlayAgent(
        state_dim=STATE_SIZE,
        action_dim=ACTION_SPACE_SIZE,
        model_path="dqn_policy/dqn_policy_best_ep1500.pt",
    )
    random_agent = RandomAgent()
    # Play one visualized game with random first player
    agents = [dqn_agent, random_agent]
    random.shuffle(agents)
    print(
        f"Player 1: {'DQN' if isinstance(agents[0], DQNPlayAgent) else 'RandomAgent'}"
    )
    print(
        f"Player 2: {'DQN' if isinstance(agents[1], DQNPlayAgent) else 'RandomAgent'}"
    )
    play_game(agents[0], agents[1], visualize=True)
    # Simulate 1000 games with random first/second for each
    wins = [0, 0]
    for i in range(1000):
        agents = [dqn_agent, random_agent]
        random.shuffle(agents)
        winner = play_game(agents[0], agents[1])
        # Map winner to correct agent
        if winner is not None:
            wins[agents[winner] == dqn_agent] += 1
    print(f"DQN wins: {wins[1]}, RandomAgent wins: {wins[0]}")
