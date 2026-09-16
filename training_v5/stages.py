"""Staged arena curriculum for tag rounds, advanced by the seeker's measured find rate.

The stages only change which arenas are sampled (size, prop count, play
length); neither agent is told anything. Small, short rounds come first: a
seeker that cannot find a hider in an 8 m arena within 15 s learns nothing
from a 12 m arena and 60 s. The next stage opens once the seeker has seen the
hider at least once in most of the recent rounds.
"""
from protocol import ARENA_SIZES, BOX_RANGE, RAMP_RANGE, LONG_PLAY, LONG_PLAY_WEIGHTS

STAGES = (
    dict(name='small', sizes=(8, 9), boxes=(2, 6), ramps=(0, 2), play=(188,), weights=(1.,)),
    dict(name='medium', sizes=(8, 9, 10), boxes=(2, 8), ramps=RAMP_RANGE, play=(188, 375), weights=(.5, .5)),
    dict(name='full', sizes=tuple(ARENA_SIZES), boxes=BOX_RANGE, ramps=RAMP_RANGE, play=LONG_PLAY, weights=tuple(LONG_PLAY_WEIGHTS)),
)
WINDOW = 256              # recent rounds that decide an advance
THRESHOLD = .8            # seeker find rate needed to open the next stage
MINIMUM_EPISODES = 2048   # rounds at a stage before it can be left


class Progression:
    """Current stage plus the recent find-rate evidence, saved in every checkpoint."""

    def __init__(self, state=None, enabled=True):
        self.enabled = bool(enabled)
        self.stage = int(state['stage']) if state else (0 if self.enabled else len(STAGES) - 1)
        self.episodes = int(state['episodes']) if state else 0
        self.found = [bool(value) for value in state['found']] if state else []
        if not 0 <= self.stage < len(STAGES):
            raise ValueError('Unknown curriculum stage')

    @property
    def name(self):
        return STAGES[self.stage]['name']

    def scope(self):
        return STAGES[self.stage]

    def ready(self):
        if not self.enabled or self.stage >= len(STAGES) - 1:
            return False
        if self.episodes < MINIMUM_EPISODES or len(self.found) < WINDOW:
            return False
        return sum(self.found) / len(self.found) >= THRESHOLD

    def observe(self, found):
        """Record one finished round; True when the stage advanced."""
        if not self.enabled:
            return False
        self.found = (self.found + [bool(found)])[-WINDOW:]
        self.episodes += 1
        if not self.ready():
            return False
        self.stage += 1
        self.found = []
        self.episodes = 0
        return True

    def find_rate(self):
        return sum(self.found) / len(self.found) if self.found else None

    def state_dict(self):
        return dict(stage=self.stage, episodes=self.episodes, found=list(self.found), enabled=self.enabled)
