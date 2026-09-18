"""Training-map replay by observed outcome progress, diversity and staleness.

This is a progress-based curriculum, not an implementation of PLR's TD-error
estimator. Every visit collects fresh on-policy actions. Evaluation maps never
enter this archive; training seeds occupy a separate range.
"""
import copy
import json
import numpy as np

CAPACITY = 384
FRESH_SHARE = .5
UNIFORM_SHARE = .25


class LevelReplay:
    def __init__(self, state=None):
        self.entries = copy.deepcopy(state['entries']) if state else {}
        self.clock = int(state['clock']) if state else 0

    @staticmethod
    def identity(config):
        if not 0 <= int(config['seed']) < 2 ** 30:
            raise ValueError('Only training seeds may enter the map curriculum')
        return json.dumps(config, sort_keys=True)

    def observe(self, config, hidden_fraction):
        key = self.identity(config)
        value = float(np.clip(hidden_fraction, 0, 1))
        self.clock += 1
        if key not in self.entries:
            self.entries[key] = dict(config=copy.deepcopy(config), fast=value, slow=value,
                                     count=0, last=self.clock)
        row = self.entries[key]
        row['fast'] += .2 * (value - row['fast'])
        row['slow'] += .03 * (value - row['slow'])
        row['count'] += 1
        row['last'] = self.clock
        if len(self.entries) > CAPACITY:
            # Evict from a crowded family/play stratum rather than losing rare layouts.
            def stratum(r):
                return r['config']['scenario'], r['config']['play']
            strata = [stratum(r) for r in self.entries.values()]
            crowded = max(set(strata), key=lambda x: (strata.count(x), x))
            victim = min((k for k, r in self.entries.items() if stratum(r) == crowded),
                         key=lambda k: (self.entries[k]['last'], k))
            del self.entries[victim]

    def sample(self, rng, fresh_factory):
        if not self.entries or rng.random() < FRESH_SHARE:
            return fresh_factory()
        strata = sorted({(r['config']['scenario'], r['config']['play']) for r in self.entries.values()})
        chosen = strata[int(rng.integers(len(strata)))]
        rows = [r for r in self.entries.values() if (r['config']['scenario'], r['config']['play']) == chosen]
        scores = np.array([.05 + abs(r['fast'] - r['slow']) + .1 / np.sqrt(r['count'])
                           + .1 * min(1, (self.clock - r['last']) / CAPACITY) for r in rows])
        p = (1 - UNIFORM_SHARE) * scores / scores.sum() + UNIFORM_SHARE / len(rows)
        return copy.deepcopy(rows[int(rng.choice(len(rows), p=p))]['config'])

    def state_dict(self):
        return copy.deepcopy(dict(entries=self.entries, clock=self.clock))
