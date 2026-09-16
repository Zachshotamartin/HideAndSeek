"""Widen a 210-input actor to a wider observation schema without changing what it computes.

New input columns (the last-seen memory, the time remaining and the observed
exploration noise) get zero weights, the two button columns move to their new
place, and every other parameter is copied. The widened actor therefore
produces exactly the original outputs until training moves the new weights.
"""
import copy

import torch

from entity_actor import ENTITY, FIXED_INDICES, LEGACY, TAG_FORMAT, build_actor, load_pair, fixed_indices
from game import NOISE, SCHEMA
from persistent_actor import OBSERVATIONS as LEGACY_SIZE, PHYSICS_OBSERVATIONS


def physical_size(observation_size, noise_rho):
    return int(observation_size) - 2 - (NOISE if noise_rho else 0)


def source_column(index, new_physical, old_physical=PHYSICS_OBSERVATIONS):
    """The legacy observation column that feeds new column ``index``; None for a new input."""
    if index < old_physical:
        return index
    if new_physical <= index < new_physical + 2:
        return old_physical + index - new_physical
    return None


def widen_columns(weight, mapping):
    """A (rows, len(mapping)) matrix: copied where mapped, zero for new inputs."""
    columns = [weight[:, source] if source is not None else torch.zeros_like(weight[:, 0]) for source in mapping]
    return torch.stack(columns, dim=1).contiguous()


def widen_state(state, kind, observation_size, noise_rho):
    new_physical = physical_size(observation_size, noise_rho)
    if new_physical < PHYSICS_OBSERVATIONS:
        raise ValueError('The widened schema must keep the 208 physical measurements')
    result = {key: value.clone() for key, value in state.items()}
    if kind == LEGACY:
        mapping = [source_column(index, new_physical) for index in range(observation_size)]
        result['encoder.weight'] = widen_columns(state['encoder.weight'], mapping)
        return result
    mapping = []
    for index in fixed_indices(observation_size):
        source = source_column(index, new_physical)
        mapping.append(FIXED_INDICES.index(source) if source is not None else None)
    for key in ('encoder.fixed.weight', 'encoder.query.weight'):
        result[key] = widen_columns(state[key], mapping)
    return result


def widen_models(models, kinds, observation_size, noise_rho):
    widened = []
    for model, kind in zip(models, kinds):
        embedding = model.encoder.embedding_size if kind == ENTITY else None
        target = build_actor(kind, model.hidden_size, model.encoder_size, embedding, observation_size, noise_rho)
        target.load_state_dict(widen_state(model.state_dict(), kind, observation_size, noise_rho))
        widened.append(target)
    return widened


def widen_record(record, observation_size, noise_rho, learning_rate=.0001):
    """A copy of a pair record on the wider schema, with fresh actor optimizers."""
    models, _ = load_pair(record, frozen=False)
    if any(model.observation_size != LEGACY_SIZE for model in models):
        raise ValueError('Only the original 210-input pair can be widened')
    kinds = [ENTITY if kind == ENTITY else LEGACY for kind in record.get('encoderTypes', [LEGACY, LEGACY])]
    widened = widen_models(models, kinds, observation_size, noise_rho)
    result = copy.deepcopy(record)
    result.update(
        format=TAG_FORMAT, encoderTypes=kinds, observationSize=int(observation_size),
        physicsObservationSize=physical_size(observation_size, noise_rho), noiseRho=float(noise_rho), schema=SCHEMA,
        models=[model.state_dict() for model in widened],
        optimizers=[torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5).state_dict() for model in widened],
        newActorUpdates=0)
    result['provenance'] = dict(record['provenance'], schemaMigration=dict(
        fromObservationSize=LEGACY_SIZE, toObservationSize=int(observation_size), noiseRho=float(noise_rho), schema=SCHEMA,
        description='Zero weights on the new last-seen, time-remaining and noise inputs; button columns moved; '
                    'identical outputs at migration; fresh actor optimizers.'))
    return result
