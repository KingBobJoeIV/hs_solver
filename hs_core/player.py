from typing import List, Optional
from .card import Card
from .minion import Minion
from .utils import parse_effect
import random


class Player:
    def __init__(self, name: str, deck: List[Card], is_first: bool):
        self.name = name
        self.hero_hp = 5
        self.max_mana = 0
        self.mana = 0
        self.deck = deck[:]
        self.hand: List[Card] = []
        self.board: List[Minion] = []
        self.fatigue = 0
        self.is_first = is_first
        self.has_coin = not is_first
        self.turn = 0

    def draw(self):
        if not self.deck:
            self.fatigue += 1
            self.hero_hp -= self.fatigue
            return None
        if len(self.hand) < 4:
            card = self.deck.pop(0)
            self.hand.append(card)
            return card
        else:
            self.deck.pop(0)
            return None

    def mulligan(self, indices: List[int]):
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self.hand):
                self.deck.append(self.hand.pop(idx))

        random.shuffle(self.deck)
        while len(self.hand) < (1 if self.is_first else 2):
            self.draw()

    def start_turn(self):
        self.turn += 1
        self.max_mana = min(5, self.turn)
        self.mana = self.max_mana
        self.draw()
        for minion in self.board:
            minion.can_attack = True
        if self.has_coin and any(c.name == "The Coin" for c in self.hand):
            pass

    def play_card(self, card_idx: int, target: Optional[str], opponent):
        card = self.hand[card_idx]
        if card.mana > self.mana:
            return False
        if card.card_type == "minion":
            if len(self.board) >= 3:
                return False
            self.mana -= card.mana
            minion = Minion(card)
            self.board.append(minion)
            effect_type, amount = parse_effect(card.effect)
            if effect_type == "deal" and amount is not None:
                if target == "hero":
                    opponent.hero_hp -= amount
                elif target is not None and target.isdigit():
                    idx = int(target)
                    if 0 <= idx < len(opponent.board):
                        opponent.board[idx].health -= amount
            self.hand.pop(card_idx)
            return True
        elif card.card_type == "spell":
            effect_type, amount = parse_effect(card.effect)
            if effect_type == "deal" and amount is not None:
                if target == "hero":
                    opponent.hero_hp -= amount
                elif target is not None and target.isdigit():
                    idx = int(target)
                    if 0 <= idx < len(opponent.board):
                        opponent.board[idx].health -= amount
            elif card.effect == "coin":
                self.mana += 1
            self.mana -= card.mana
            self.hand.pop(card_idx)
            return True
        return False

    def attack_with_minion(self, minion_idx: int, target: str, opponent):
        minion = self.board[minion_idx]
        if not minion.can_attack:
            return False
        if target == "hero":
            opponent.hero_hp -= minion.attack
        elif target.isdigit():
            idx = int(target)
            if 0 <= idx < len(opponent.board):
                opponent.board[idx].health -= minion.attack
                minion.health -= opponent.board[idx].attack
        minion.can_attack = False
        return True

    def cleanup(self):
        self.board = [m for m in self.board if m.health > 0]
