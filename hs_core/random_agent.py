import random


class RandomAgent:
    def choose_action(self, state):
        actions = state.get_legal_actions()
        return random.choice(actions)
