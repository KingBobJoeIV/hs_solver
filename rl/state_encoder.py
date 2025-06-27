import numpy as np


def encode_state(game_state, player_idx):
    """
    Encode the game state into a flat numpy array for DQN.
    This is a simple encoding. You may want to improve it for better learning.
    """
    player = game_state.players[player_idx]
    opponent = game_state.players[1 - player_idx]
    state = []
    # Hero HP, Mana, Deck, Fatigue
    state += [
        player.hero_hp,
        player.mana,
        player.max_mana,
        len(player.deck),
        player.fatigue,
    ]
    state += [
        opponent.hero_hp,
        opponent.mana,
        opponent.max_mana,
        len(opponent.deck),
        opponent.fatigue,
    ]
    # Hand encoding (one-hot per card type, up to 4 cards)
    card_names = [
        "1/1",
        "1 mana deal 1",
        "2/2",
        "2 1/1 deal 1",
        "3/3",
        "3 2/2 deal 1",
        "3 mana deal 2",
        "The Coin",
    ]
    hand_vec = np.zeros((4, len(card_names)))
    for i, card in enumerate(player.hand[:4]):
        if card.name in card_names:
            hand_vec[i, card_names.index(card.name)] = 1
    state += hand_vec.flatten().tolist()
    # Board encoding (attack, health for up to 3 minions)
    for minion in player.board[:3]:
        state += [minion.attack, minion.health, int(minion.can_attack)]
    for _ in range(3 - len(player.board)):
        state += [0, 0, 0]
    for minion in opponent.board[:3]:
        state += [minion.attack, minion.health, int(minion.can_attack)]
    for _ in range(3 - len(opponent.board)):
        state += [0, 0, 0]
    return np.array(state, dtype=np.float32)
