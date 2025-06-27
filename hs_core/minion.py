from .card import Card


class Minion:
    def __init__(self, card: Card):
        self.card = card
        self.attack = card.attack
        self.health = card.health
        self.can_attack = False

    def __repr__(self):
        return f"{self.card.name} ({self.attack}/{self.health})"
