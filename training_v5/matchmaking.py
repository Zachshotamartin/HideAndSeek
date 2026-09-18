"""Outcome-based historical hider sampling; no routes or tool strategies."""
import numpy as np

UNIFORM_SHARE = .25
UPDATE_RATE = .1


class SeekerMatchmaker:
    def __init__(self, count, state=None):
        self.success = list(state['success']) if state else [.5] * count
        self.counts = list(state['counts']) if state else [0] * count
        if len(self.success) != count or len(self.counts) != count:
            raise ValueError('League and matchmaking state differ')

    def probabilities(self):
        p = np.asarray(self.success)
        # Intermediate success is learnable, but every hard/easy opponent keeps
        # a nonzero uniform share. Unknown opponents begin at intermediate success.
        weights = .05 + 4 * p * (1 - p)
        return (1 - UNIFORM_SHARE) * weights / weights.sum() + UNIFORM_SHARE / len(p)

    def assign(self, roles, rng):
        result = roles.copy()
        rows = np.flatnonzero((roles[:, 0] >= 0) & (roles[:, 1] == -1))
        if len(rows):
            result[rows, 0] = rng.choice(len(self.success), size=len(rows), p=self.probabilities())
        return result

    def observe(self, opponent, captured):
        if opponent < 0:
            return
        self.success[opponent] += UPDATE_RATE * (float(captured) - self.success[opponent])
        self.counts[opponent] += 1

    def append(self):
        self.success.append(.5)
        self.counts.append(0)

    def prune(self, keep):
        self.success = [self.success[i] for i in keep]
        self.counts = [self.counts[i] for i in keep]

    def state_dict(self):
        return dict(success=list(self.success), counts=list(self.counts))
