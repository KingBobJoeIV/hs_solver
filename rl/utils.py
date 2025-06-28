import rl.constants as constants
import math
import numpy as np


def get_beta(episode):
    """Beta annealing for prioritized replay"""
    return min(
        constants.BETA_END,
        constants.BETA_START
        + (constants.BETA_END - constants.BETA_START) * episode / constants.EPISODES,
    )


def get_learning_rate(episode):
    """Cosine annealing learning rate schedule"""
    return constants.LR * 0.5 * (1 + math.cos(math.pi * episode / constants.EPISODES))


def enhanced_encode_state(game_state, player_idx):
    """Enhanced state encoding with normalized features and more game context"""
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
        player.turn / 10.0,  # normalized turn number
        int(player_idx == game_state.current),
        int(getattr(player, "is_first", False)),  # is first player
        int(not getattr(player, "is_first", False)),  # is second player
        (player.turn + opponent.turn) / 20.0,  # normalized total turn
    ]

    # Add has_coin feature (1 if The Coin in hand, else 0)
    state.append(
        1.0
        if any(getattr(card, "name", None) == "The Coin" for card in player.hand)
        else 0.0
    )

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
