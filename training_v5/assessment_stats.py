"""Paired cluster bootstrap. Independent seeds/worlds, never individual frames."""
import numpy as np


def paired_interval(candidate, reference, *, confidence=.95, resamples=10000, seed=714211, minimum_units=5):
    if set(candidate) != set(reference) or not candidate:
        raise ValueError('Paired observations require identical nonempty identities')
    if not 0 < confidence < 1 or resamples < 100:
        raise ValueError('Invalid bootstrap configuration')
    keys = sorted(candidate)
    a, b = [np.asarray([data[k] for k in keys], dtype=float) for data in (candidate, reference)]
    if a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('All paired metrics must be finite scalars')
    delta = a - b
    random = np.random.default_rng(seed)
    # Bounded memory even for a large evaluation cohort.
    boot = np.concatenate([random.choice(delta, (min(256, resamples - start), len(delta)), replace=True).mean(1)
                           for start in range(0, resamples, 256)])
    alpha = (1 - confidence) / 2
    interval = np.quantile(boot, [alpha, 1 - alpha]).tolist()
    return dict(units=len(delta), candidateMean=float(a.mean()), referenceMean=float(b.mean()),
                difference=float(delta.mean()), interval=interval, confidence=confidence,
                preliminary=len(delta) < minimum_units, independentUnitMinimum=minimum_units,
                method='paired percentile cluster bootstrap', perUnitDifferences=delta.tolist())


def gate(contrasts, tolerances, primary, minimum_gain):
    """Predeclared noninferiority on every guard and positive primary improvement.

    Use simultaneous intervals when testing several criteria. This never
    automatically publishes a model, and development selection is not a final test.
    """
    if set(tolerances) - set(contrasts) or primary not in contrasts:
        raise ValueError('Missing required promotion evidence')
    reasons = []
    if any(row['preliminary'] for row in contrasts.values()):
        reasons.append('too few independent units')
    for metric, tolerance in tolerances.items():
        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValueError('Noninferiority tolerances must be finite and nonnegative')
        if contrasts[metric]['interval'][0] < -tolerance:
            reasons.append(metric + ': noninferiority not established')
    gain = contrasts[primary]
    if gain['interval'][0] <= 0 or gain['difference'] < minimum_gain:
        reasons.append(primary + ': meaningful improvement not established')
    return dict(eligibleForReview=not reasons, reasons=reasons, automaticPublication=False)
