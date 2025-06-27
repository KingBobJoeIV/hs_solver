import rl.constants as constants
import math


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
