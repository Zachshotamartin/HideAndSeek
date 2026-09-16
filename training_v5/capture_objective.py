"""Explicit training-only choice of terminal capture credit, separate from game scores."""
import math
import numpy as np
from fit_central_value import GAMMA

MODES = ('immediate', 'discounted-equivalent')


def future_credit(remaining, gamma=GAMMA):
    if not isinstance(remaining, (int, np.integer)) or remaining < 0 or not 0 < gamma <= 1:
        raise ValueError('Expected nonnegative remaining steps and discount in (0,1]')
    if gamma == 1:
        return float(remaining)
    return gamma * (-math.expm1(remaining * math.log(gamma))) / (1 - gamma)


def calibrate_capture(rewards, infos, mode='immediate'):
    if mode not in MODES:
        raise ValueError('Unknown capture credit mode')
    result = np.asarray(rewards, dtype=float).copy()
    if mode == 'discounted-equivalent':
        for index, info in enumerate(infos):
            if info.get('captured', False):
                remaining = int(info['play_steps'] - info['captureStep'])
                change = future_credit(remaining) - remaining
                result[index] += [-change, change]
    return result
