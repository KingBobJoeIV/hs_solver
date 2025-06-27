from typing import Tuple
from .player import Player
from .utils import parse_effect


class GameState:
    def __init__(self, p1: Player, p2: Player):
        self.players = [p1, p2]
        self.current = 0
        self.winner = None

    def other(self):
        return 1 - self.current

    def step(self, action: Tuple):
        player = self.players[self.current]
        opponent = self.players[self.other()]
        if action[0] == "play":
            card_idx, target = action[1], action[2]
            player.play_card(card_idx, target, opponent)
        elif action[0] == "attack":
            minion_idx, target = action[1], action[2]
            player.attack_with_minion(minion_idx, target, opponent)
        elif action[0] == "end":
            self.current = self.other()
            self.players[self.current].start_turn()
        player.cleanup()
        opponent.cleanup()
        if player.hero_hp <= 0:
            self.winner = self.other()
        elif opponent.hero_hp <= 0:
            self.winner = self.current

    def is_terminal(self):
        return self.winner is not None

    def get_legal_actions(self):
        player = self.players[self.current]
        opponent = self.players[self.other()]
        actions = []
        for i, card in enumerate(player.hand):
            if card.mana <= player.mana:
                if card.card_type == "minion":
                    if len(player.board) < 3:
                        effect_type, _ = parse_effect(card.effect)
                        if effect_type == "deal":
                            actions.append(("play", i, "hero"))
                            for j in range(len(opponent.board)):
                                actions.append(("play", i, str(j)))
                        else:
                            actions.append(("play", i, None))
                elif card.card_type == "spell":
                    effect_type, _ = parse_effect(card.effect)
                    if effect_type == "deal":
                        actions.append(("play", i, "hero"))
                        for j in range(len(opponent.board)):
                            actions.append(("play", i, str(j)))
                    elif card.effect == "coin":
                        actions.append(("play", i, None))
        for i, minion in enumerate(player.board):
            if minion.can_attack:
                actions.append(("attack", i, "hero"))
                for j in range(len(opponent.board)):
                    actions.append(("attack", i, str(j)))
        actions.append(("end",))
        return actions
