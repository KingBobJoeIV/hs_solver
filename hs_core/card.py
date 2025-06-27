from typing import Optional


class Card:
    def __init__(
        self,
        name: str,
        mana: int,
        card_type: str,
        attack: int = 0,
        health: int = 0,
        effect: Optional[str] = None,
    ):
        self.name = name
        self.mana = mana
        self.card_type = card_type  # 'minion' or 'spell'
        self.attack = attack
        self.health = health
        self.effect = effect

    def __repr__(self):
        return f"{self.name}"
