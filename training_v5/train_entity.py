"""League self-play of the tag-round game with recurrent PPO.

Both encoder arms use league B, identical inherited critic weights and fresh
actor/critic optimizers. Runtime models and earlier trainers are untouched.
Workers return the game.py observation (physics plus last-seen memory and the
clock); every actor, current or frozen, takes the prefix its schema defines.
"""
import argparse
import copy
import json
import os
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch

from central_critic import FEATURES, batch_states
from central_persistent_train import advantages_and_returns
from value_learning import initialize_critic, update_critic, prediction_metrics, PopArtCritic, NORMALIZED_SCHEMA
from capture_objective import calibrate_capture
from level_replay import LevelReplay
from entity_actor import ENTITY, FORMAT, TAG_FORMAT, LEGACY, EntityActor, load_pair, restore_league
from env_pool import PhysicsEnvPool
from fit_central_value import REWARD_SCALE
from game import NOISE, SCHEMA, SEEKER_SPEED_ABLATION, scale_actions, seeker_found
from league_ppo import act_grouped, active_masks, actor_update
from persistent_train import file_hash
from matchmaking import SeekerMatchmaker
from motion_diagnostics import blocked_adjustment
from protocol import archive_indices, assign_for_variant, environment_config
from residual_critic import ResidualCentralCritic, SCHEMA
from stages import Progression

torch.set_num_threads(1)
LEGACY_OBSERVATION_SIZE = 210
ACTIONS = 6
DESCRIPTOR_SIZE = 10            # mean |action| (6), path per step (2), grabs per step (2)
RECENT_EPISODES = 256
NEWEST_SNAPSHOTS = 8
MINIMUM_ARCHIVE = 13            # anchors, eight newest and at least one diverse pick
SEED_LIMIT = 2 ** 30
CRITIC_STATE_OFFSET = 228
SEEKER_ACTIVE_FRACTION = .8
RESUME_FIELDS = ['encoder', 'arm', 'envs', 'horizon', 'sequence_length', 'burn_in', 'snapshot_every', 'archive_limit',
                 'sequence_batch', 'critic_batch_size', 'epochs', 'learning_rate', 'critic_learning_rate', 'entropy',
                 'kl_limit', 'seed', 'variant', 'matchmaking', 'blocked_cost', 'value_normalization', 'map_replay', 'capture_credit',
                 'noise_rho', 'curriculum']
SOURCE_NAMES = ['train_entity.py', 'entity_actor.py', 'actor.py', 'league_ppo.py', 'central_persistent_train.py',
                'residual_critic.py', 'central_critic.py', 'persistent_actor.py', 'persistent_train.py',
                'fit_central_value.py', 'env_pool.py', 'physics.py', 'protocol.py', 'snapshots.py', 'capture.py', 'matchmaking.py',
                'motion_diagnostics.py', 'value_learning.py', 'capture_objective.py', 'level_replay.py', 'game.py', 'stages.py',
                'widen_schema.py']
RECENT_ONLY_VARIANTS = ('baseline', 'recent-only')


def load_actors(saved, frozen=False):
    return load_pair(saved, frozen=frozen)[0]


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


# --------------------------------------------------------------- validation
def validate_shape(args):
    blocked_adjustment([False, False], args.blocked_cost)
    if not 0 <= args.noise_rho < 1:
        raise ValueError('The exploration noise correlation must lie in [0, 1)')
    if args.horizon % args.sequence_length or args.envs < 1:
        raise ValueError('Positive environment count and complete recurrent sequences required')
    if args.burn_in < 0 or args.burn_in > args.horizon or args.snapshot_every < 1 or args.archive_limit < MINIMUM_ARCHIVE:
        raise ValueError('Burn-in must fit the horizon; the archive limit must leave room beyond anchors and the eight newest snapshots')


def validate_protocol(args, protocol):
    """Every predeclared comparison setting and asset must match the command line."""
    for field, expected in protocol['training'].items():
        if getattr(args, field) != expected:
            raise ValueError(f'Predeclared comparison setting changed: {field}')
    for field, asset in [('parent', args.encoder), ('critic', 'critic'), ('initial', 'initial')]:
        if file_hash(getattr(args, field)) != protocol['assets'][asset]['sha256']:
            raise ValueError(f'Predeclared comparison asset changed: {field}')
    if [file_hash(path) for path in args.history] != [protocol['assets'][name]['sha256'] for name in protocol['history']]:
        raise ValueError('Predeclared frozen opponent pool changed')


def validate_assets(args, protocol, parent, critic_source, initial, physics_hash):
    if initial['decisions'] != 0:
        raise ValueError('The fixed initial comparison must be a compatible zero-experience policy')
    if physics_hash != protocol['physicsSHA256']:
        raise ValueError('Predeclared physics changed')
    expected_type = ENTITY if args.encoder == 'entity' else LEGACY
    if parent.get('encoderTypes') != [expected_type] * 2 or parent.get('newActorUpdates') != 0:
        raise ValueError('Use the matching immutable zero-update prepared arm')
    if any(state['state'] for state in parent['optimizers']):
        raise ValueError('Both arms require equally empty initial actor Adam states')
    if critic_source['provenance']['physicsSHA256'] != physics_hash:
        raise ValueError('Inherited critic must belong to this same physical game')
    if args.arm != 'B':
        raise ValueError('Both encoder arms use the same frozen-opponent league B')
    if parent['provenance']['physicsSHA256'] != physics_hash:
        raise ValueError('Versioned physics identity changed')


def validate_resume(args, resumed, parent_hash, history_hashes):
    if resumed['provenance']['parentSHA256'] != parent_hash or resumed['arguments']['encoder'] != args.encoder:
        raise ValueError('Resume requires the same parent and trial arm')
    if resumed['provenance']['protocolSHA256'] != file_hash(args.protocol):
        raise ValueError('Resume requires the identical predeclared protocol')
    if [entry['sha256'] for entry in resumed['provenance']['history']] != history_hashes:
        raise ValueError('Resume requires the identical ordered frozen opponent pool')
    if resumed['provenance']['criticSourceSHA256'] != file_hash(args.critic):
        raise ValueError('Resume requires the identical initial critic asset')
    if resumed['provenance']['initialSHA256'] != file_hash(args.initial):
        raise ValueError('Resume requires the identical fixed initial comparison')
    for field in RESUME_FIELDS:
        if resumed['arguments'][field] != getattr(args, field):
            raise ValueError(f'Cannot silently change resumed setting: {field}')


def fresh_provenance(args, protocol, parent, parent_hash, physics_hash, history_paths, history_records, models):
    return dict(
        parentSHA256=parent_hash, parentDecisions=parent['decisions'], protocolSHA256=file_hash(args.protocol),
        sourceSelectedPairSHA256=parent['provenance']['sourceSelectedPairSHA256'],
        inheritedRoleSources=parent['provenance']['roleSources'],
        encoderPreparation=parent['provenance']['encoderPreparation'],
        physicsSHA256=physics_hash, criticSourceSHA256=file_hash(args.critic),
        initialSHA256=file_hash(args.initial), criticWarmupDecisions=0,
        criticInitialization=('Fresh central critic, matched across initialization arms; new AdamW state.' if parent['provenance'].get('initializationComparison') else 'Same inherited residual critic weights in both arms; new AdamW state. No claim of calibration to the projected actor.'),
        optimizerInitialization='Both actor Adam states and central critic AdamW state start empty in both arms.',
        history=[dict(file=str(path.resolve()), sha256=file_hash(path), decisions=record['decisions'])
                 for path, record in zip(history_paths, history_records)],
        actorInput=actor_input_description(models[0]),
        rule=protocol['reward'],
        trainingOptions=dict(valueNormalization=args.value_normalization, mapReplay=args.map_replay,
                             captureCredit=args.capture_credit, blockedCost=args.blocked_cost, noiseRho=args.noise_rho,
                             curriculum=args.curriculum, seekerSpeed=seeker_speed(args.variant)),
        comparisons=protocol.get('comparisons', 'Same league B and unchanged visibility reward.'),
        architecture=[dict(memory=m.hidden_size, encoder=m.encoder_size, parameters=sum(p.numel() for p in m.parameters()))
                      for m in models],
        recurrence=dict(sequenceLength=args.sequence_length, burnIn=args.burn_in,
                        description='Truncated backpropagation over complete sequences; burn-in re-runs the recurrence through the previous rollout tail without gradients.'),
        archive=dict(snapshotEvery=args.snapshot_every, limit=args.archive_limit),
        sourceHistory=[], resumes=[])


def actor_input_description(model):
    if model.observation_size == LEGACY_OBSERVATION_SIZE:
        return 'Restricted 210 observations; all ten object slots and thirty range sensors; no opponent identity or central state'
    return (f'{model.observation_size} observations: 208 physical measurements, last-seen opponent offset and age, '
            f'time remaining, two own buttons and {NOISE} observed exploration-noise values; no opponent identity or central state')


def seeker_speed(variant):
    return SEEKER_SPEED_ABLATION if variant == 'slow-seeker' else 1.


class Trainer:
    """One league self-play run: validated inputs, live rollout state and the update loop."""

    def __init__(self, args, stop_requested=None, on_checkpoint=None):
        if not hasattr(args, 'variant'):
            args.variant = 'full'
        self.args = args
        self.stop_requested = stop_requested
        self.on_checkpoint = on_checkpoint
        validate_shape(args)
        self.block = args.envs * args.horizon * 2
        self.target_updates = args.target_interactions // self.block
        if not self.target_updates:
            raise ValueError('Interaction target is smaller than one complete rollout')
        self.protocol = json.loads(Path(args.protocol).read_text())
        validate_protocol(args, self.protocol)
        self.parent = load_checkpoint(args.parent)
        self.critic_source = load_checkpoint(args.critic)
        self.initial = load_checkpoint(args.initial)
        self.parent_hash = file_hash(args.parent)
        self.physics_hash = file_hash(Path(__file__).with_name('physics.py'))
        validate_assets(args, self.protocol, self.parent, self.critic_source, self.initial, self.physics_hash)
        self.load_league()
        self.resumed = load_checkpoint(args.resume) if args.resume else None
        if self.resumed:
            validate_resume(args, self.resumed, self.parent_hash, self.history_hashes)
        if self.resumed and 'leagueModels' in self.resumed:
            self.histories = restore_league(self.resumed)
            self.frozen_before = [[copy.deepcopy(m.state_dict()) for m in pair] for pair in self.histories]
        self.load_learners()
        self.world_generator = np.random.default_rng(args.seed)
        self.opponent_generator = np.random.default_rng(args.seed + 1)
        torch.set_rng_state((self.resumed or self.parent)['torchRNG'])
        if self.resumed:
            self.world_generator.bit_generator.state = self.resumed['worldRNG']
            self.opponent_generator.bit_generator.state = self.resumed['opponentRNG']
        self.prepare_destination()
        self.restore_counters()
        self.seeker_speed = seeker_speed(args.variant)
        self.progression = Progression((self.resumed or {}).get('curriculumState'), enabled=args.curriculum == 'staged')
        self.configs = [dict(seed=int(self.world_generator.integers(1, SEED_LIMIT)),
                             **self.environment(index)) for index in range(args.envs)]
        self.descriptors = list(self.resumed.get('leagueDescriptors', [])) if self.resumed else []
        if len(self.descriptors) != len(self.histories):
            self.descriptors = [[0.] * DESCRIPTOR_SIZE for _ in self.histories]
        self.role_ids = assign_for_variant(self.opponent_generator, args.envs, len(self.histories), SEEKER_ACTIVE_FRACTION, args.variant)
        self.matchmaker = SeekerMatchmaker(len(self.histories), (self.resumed or {}).get("matchmakingState"))
        self.level_replay = LevelReplay((self.resumed or {}).get("levelReplayState"))
        if args.matchmaking == "seeker-curriculum":
            self.role_ids = self.matchmaker.assign(self.role_ids, self.opponent_generator)
        self.burn_in = int(args.burn_in)
        self.prefixes = [None, None]
        if self.resumed and 'rolloutState' in self.resumed:
            self.configs = self.resumed['rolloutState']['pool']['configs']
        self.retained = {int(value) for value in args.retain_updates.split(',') if value}
        self.pool = None

    # ----------------------------------------------------------------- setup
    def load_league(self):
        self.history_paths = [Path(path) for path in self.args.history]
        self.history_records = [load_checkpoint(path) for path in self.history_paths]
        if not self.history_records or any(record['provenance']['physicsSHA256'] != self.physics_hash for record in self.history_records):
            raise ValueError('Historical opponents must use this unchanged physical game')
        self.histories = [load_actors(record, frozen=True) for record in self.history_records]
        self.history_hashes = [file_hash(path) for path in self.history_paths]
        self.frozen_before = [[copy.deepcopy(model.state_dict()) for model in pair] for pair in self.histories]

    def load_learners(self):
        args, resumed = self.args, self.resumed
        self.models = load_actors(resumed or self.parent)
        self.hidden_size = self.models[0].hidden_size
        self.observation_size = self.models[0].observation_size
        if self.models[1].hidden_size != self.hidden_size or any(m.hidden_size > self.hidden_size for pair in self.histories for m in pair):
            raise ValueError('Current actors need equal memory capacity, no smaller than historical opponents')
        if self.models[1].observation_size != self.observation_size or any(m.noise_rho != args.noise_rho for m in self.models):
            raise ValueError('Both current actors must share the observation schema and the declared noise correlation')
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5) for model in self.models]
        if resumed:
            for optimizer, state in zip(self.optimizers, resumed['optimizers']):
                optimizer.load_state_dict(state)
        self.critic, self.critic_optimizer = initialize_critic(self.critic_source, args.critic_learning_rate, resumed, args.value_normalization)

    def prepare_destination(self):
        """Create the output folder, record provenance and copy every training source into it."""
        args, resumed = self.args, self.resumed
        self.destination = Path(args.output)
        self.destination.mkdir(parents=True, exist_ok=True)
        if (self.destination / 'latest.pt').exists() and not resumed:
            raise ValueError('Do not overwrite an existing comparison arm')
        if not resumed:
            shutil.copyfile(args.parent, self.destination / 'parent.pt')
            shutil.copyfile(args.initial, self.destination / 'initial.pt')
        sources = {name: file_hash(Path(__file__).with_name(name)) for name in SOURCE_NAMES}
        if resumed:
            self.provenance = copy.deepcopy(resumed['provenance'])
        else:
            self.provenance = fresh_provenance(args, self.protocol, self.parent, self.parent_hash, self.physics_hash,
                                               self.history_paths, self.history_records, self.models)
        if resumed and resumed['provenance']['sourceHistory'][-1] != sources:
            raise ValueError('Training source changed; exact resume refused. Create an explicit new protocol instead.')
        self.provenance['sourceHistory'].append(sources)
        if resumed:
            self.provenance['resumes'].append(dict(
                afterInteractions=resumed['totalPolicyInteractions'],
                worldsRestarted=0 if 'rolloutState' in resumed else args.envs,
                reason='Full physical/weld/RNG/memory restoration when rolloutState is present; legacy checkpoints explicitly reset worlds.'))
        source_dir = self.destination / 'source' / str(len(self.provenance['sourceHistory']))
        source_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.protocol, source_dir / 'PROTOCOL.json')
        for name in SOURCE_NAMES:
            shutil.copyfile(Path(__file__).with_name(name), source_dir / name)
        (source_dir / 'SHA256.json').write_text(json.dumps(sources, indent=2) + '\n')

    def restore_counters(self):
        resumed = self.resumed
        self.first_update = resumed['pilotUpdates'] if resumed else 0
        self.interactions = resumed['totalPolicyInteractions'] if resumed else 0
        self.current_counts = np.array(resumed['currentPolicyDecisions'] if resumed else [0, 0], np.int64)
        self.historical_counts = np.array(resumed['historicalPolicyDecisions'] if resumed else [0, 0], np.int64)
        self.active_counts = np.array(resumed['activePolicySamples'] if resumed else [0, 0], np.int64)
        self.episodes = resumed['pilotEpisodes'] if resumed else 0
        self.previous_seconds = resumed['seconds'] if resumed else 0
        self.log = list(resumed['log']) if resumed else []
        self.recent = []

    def environment(self, index=0):
        return environment_config(self.world_generator, index, self.args.variant, self.progression.scope())

    def restore_rollout_state(self, resumed):
        rs = resumed['rolloutState']
        self.physical = self.pool.restore(rs['pool'])
        self.buttons = rs['buttons']
        self.noises = rs.get('noises', np.zeros((self.args.envs, 2, NOISE), np.float32))
        self.memories = rs['memories']
        self.starts = rs['starts']
        self.role_ids = rs['roleIds']
        self.prefixes = list(rs.get('burnInPrefix', [None, None]))
        self.recent = list(rs.get('recent', []))
        self.world_generator.bit_generator.state = resumed['worldRNG']
        self.opponent_generator.bit_generator.state = resumed['opponentRNG']
        torch.set_rng_state(resumed['torchRNG'])

    # ------------------------------------------------------------------ run
    def train(self):
        args = self.args
        with PhysicsEnvPool(self.configs, workers=args.workers, with_central_state=True,
                            ignore_parent_signals=getattr(args, 'graceful_worker_signals', False), observation='game') as pool:
            self.pool = pool
            self.physical = pool.observations.copy()
            self.buttons = np.zeros((args.envs, 2, 2), np.float32)
            self.noises = np.zeros((args.envs, 2, NOISE), np.float32)
            self.memories = [torch.zeros(args.envs, self.hidden_size), torch.zeros(args.envs, self.hidden_size)]
            self.starts = torch.ones(args.envs)
            if self.resumed and 'rolloutState' in self.resumed:
                self.restore_rollout_state(self.resumed)
            self.started = time.monotonic()
            for update in range(self.first_update, self.target_updates):
                if self.update(update):
                    break

    def update(self, update):
        """One rollout, both actor updates, the critic update and league upkeep; True when stopping."""
        args = self.args
        self.critic.eval()
        batch, stats = self.rollout()
        bootstrap = self.bootstrap()
        actor_stats, value_stats, optimizer_seconds = self.optimize(batch, bootstrap)
        self.interactions += self.block
        assert int(self.current_counts.sum() + self.historical_counts.sum()) == self.interactions
        row = dict(encoder=args.encoder, arm=args.arm, update=update + 1, totalPolicyInteractions=self.interactions,
                   currentPolicyDecisions=self.current_counts.tolist(), historicalPolicyDecisions=self.historical_counts.tolist(),
                   activePolicySamples=self.active_counts.tolist(), episodes=self.episodes,
                   seconds=self.previous_seconds + time.monotonic() - self.started,
                   rolloutSeconds=stats['seconds'], optimizerSeconds=optimizer_seconds,
                   hider=actor_stats[0], seeker=actor_stats[1], critic=value_stats,
                   meanHiddenFraction=float(np.mean(self.recent)) if self.recent else None,
                   unscaledRewardSum=stats['unscaled'].tolist(), divergedEpisodes=stats['diverged'],
                   curriculum=dict(stage=self.progression.name, seekerFindRate=self.progression.find_rate()))
        if (update + 1) % args.snapshot_every == 0:
            self.snapshot_league(stats)
        self.prune_league()
        row['leagueSize'] = len(self.histories)
        self.log.append(row)
        print(json.dumps(row), flush=True)
        stopping = bool(args.stop_after_updates and update + 1 >= args.stop_after_updates)
        stopping |= bool(self.stop_requested and self.stop_requested())
        if (update + 1) % args.save_every == 0 or update + 1 in self.retained or stopping or update + 1 == self.target_updates:
            self.checkpoint(update, row)
        return stopping

    # -------------------------------------------------------------- rollout
    def rollout(self):
        """Collect one horizon from every environment under the current actors and league."""
        h, n, hidden = self.args.horizon, self.args.envs, self.hidden_size
        batch = dict(
            actor_buffers=[[torch.zeros(h, n, self.observation_size), torch.zeros(h, n, hidden), torch.zeros(h, n),
                            torch.zeros(h, n, ACTIONS), torch.zeros(h, n)] for _ in range(2)],
            current_rows=torch.zeros(h, n, 2, dtype=torch.bool),
            central={key: torch.zeros(h, n, *shape) for key, shape in FEATURES.items()},
            before_memories=torch.zeros(h, n, 2, hidden), partial_values=torch.zeros(h, n, 2),
            values=torch.zeros(h, n), rewards=torch.zeros(h, n), dones=torch.zeros(h, n),
            play_lengths=torch.zeros(h, n, dtype=torch.int64))
        stats = dict(unscaled=np.zeros(2), behavior=np.zeros(DESCRIPTOR_SIZE), behavior_count=0, diverged=0)
        started = time.monotonic()
        for step in range(h):
            self.rollout_step(step, batch, stats)
        stats['seconds'] = time.monotonic() - started
        return batch, stats

    def rollout_step(self, step, batch, stats):
        pool = self.pool
        batch['play_lengths'][step] = torch.tensor([c.get('play', 144) for c in pool.configs])
        for role in range(2):
            self.memories[role] *= 1 - self.starts[:, None]
        self.noises *= 1 - self.starts.numpy()[:, None, None]
        before = torch.stack(self.memories, dim=1).clone()
        state = batch_states(pool.central_states)
        for key in FEATURES:
            batch['central'][key][step] = state[key]
        batch['before_memories'][step] = before
        current, active = active_masks(self.physical, self.role_ids)
        batch['current_rows'][step] = torch.from_numpy(current)
        self.current_counts += current.sum(0)
        self.historical_counts += (~current).sum(0)
        self.active_counts += active.sum(0)
        with torch.no_grad():
            actions, next_memories, next_buttons, next_noises, records = act_grouped(
                self.models, self.histories, self.physical, self.memories, self.buttons, self.role_ids, noises=self.noises)
            batch['partial_values'][step] = torch.stack([record[3] for record in records], dim=-1)
            batch['values'][step] = self.critic(state, before, batch['partial_values'][step])
        for role in range(2):
            observed, raw, logp, _ = records[role]
            buffers = batch['actor_buffers'][role]
            buffers[0][step] = observed
            buffers[1][step] = self.memories[role]
            buffers[2][step] = self.starts
            buffers[3][step] = raw
            buffers[4][step] = logp
        stats['behavior'][:6] += np.mean(np.abs(actions), axis=(0, 1))
        stats['behavior_count'] += 1
        self.physical, returned, done, infos = pool.step(scale_actions(actions, self.seeker_speed))
        np.testing.assert_array_equal(returned.sum(1), np.zeros(self.args.envs))
        adjusted = calibrate_capture(returned, infos, self.args.capture_credit) + blocked_adjustment([info.get('blockedMotion', [False, False]) for info in infos],
                                                  self.args.blocked_cost)
        batch['rewards'][step] = torch.from_numpy(adjusted[:, 0].copy()) * REWARD_SCALE
        batch['dones'][step] = torch.from_numpy(done.astype(np.float32))
        stats['unscaled'] += returned.sum(0)
        self.buttons, self.memories, self.noises = next_buttons, next_memories, next_noises
        finished = np.flatnonzero(done).tolist()
        if finished:
            self.finish_episodes(finished, infos, stats)
        self.starts = torch.from_numpy(done.astype(np.float32))

    def finish_episodes(self, finished, infos, stats):
        """Record finished episodes and restart their worlds on fresh randomized layouts."""
        self.recent.extend(infos[index]['hidden'] / infos[index]['play_steps'] for index in finished)
        self.recent = self.recent[-RECENT_EPISODES:]
        self.episodes += len(finished)
        stats['diverged'] += sum(1 for index in finished if infos[index].get('diverged'))
        for index in finished:
            r = infos[index]
            stats['behavior'][6:8] += np.asarray(r['path']) / max(1, r['play_steps'])
            stats['behavior'][8:] += np.asarray(r['grabs']) / max(1, r['play_steps'])
        for index in finished:
            if self.role_ids[index, 1] == -1:
                self.matchmaker.observe(int(self.role_ids[index, 0]), infos[index].get("captured", False))
            if not infos[index].get('diverged'):
                self.progression.observe(seeker_found(infos[index]))
        if self.args.map_replay == 'progress':
            for index in finished:
                if not infos[index].get('diverged'):
                    self.level_replay.observe(self.pool.configs[index], infos[index]['hidden'] / infos[index]['play_steps'])
            def fresh():
                return dict(seed=int(self.world_generator.integers(1, SEED_LIMIT)), **self.environment())
            changes = [self.level_replay.sample(self.world_generator, fresh) for _ in finished]
            seeds = [row['seed'] for row in changes]
        else:
            seeds = [int(self.world_generator.integers(1, SEED_LIMIT)) for _ in finished]
            changes = [self.environment(index) for index in finished]
        self.physical[finished] = self.pool.reset_at(finished, seeds, changes)
        self.buttons[finished] = 0
        self.noises[finished] = 0
        self.role_ids[finished] = assign_for_variant(self.opponent_generator, len(finished), len(self.histories),
                                                     self.active_counts[1] / max(1, self.current_counts[1]), self.args.variant)
        if self.args.matchmaking == "seeker-curriculum":
            self.role_ids[finished] = self.matchmaker.assign(self.role_ids[finished], self.opponent_generator)

    def bootstrap(self):
        with torch.no_grad():
            bootstrap_memories = [memory * (1 - self.starts[:, None]) for memory in self.memories]
            _, _, _, _, records = act_grouped(self.models, self.histories, self.physical, bootstrap_memories, self.buttons,
                                              self.role_ids, sample=False, noises=self.noises)
            partial = torch.stack([record[3] for record in records], dim=-1)
            return self.critic(batch_states(self.pool.central_states), torch.stack(bootstrap_memories, dim=1), partial)

    # ------------------------------------------------------------- learning
    def optimize(self, batch, bootstrap):
        args = self.args
        advantages, returns = advantages_and_returns(batch['rewards'], batch['values'], batch['dones'], bootstrap)
        started = time.monotonic()
        actor_stats = []
        for role, model in enumerate(self.models):
            actor_stats.append(actor_update(
                model, self.optimizers[role], batch['actor_buffers'][role], advantages if role == 0 else -advantages,
                batch['current_rows'][:, :, role], role, epochs=args.epochs, sequence_length=args.sequence_length,
                sequence_batch=args.sequence_batch, entropy_weight=args.entropy, kl_limit=args.kl_limit,
                burn_in=self.burn_in, prefix=self.prefixes[role]))
        # The tail of this rollout warms up the first sequence of the next one.
        if self.burn_in:
            self.prefixes = [tuple(batch['actor_buffers'][role][k][-self.burn_in:].clone() for k in range(3)) for role in range(2)]
        else:
            self.prefixes = [None, None]
        value_stats = update_critic(self.critic, self.critic_optimizer, batch['central'], batch['before_memories'],
                                    batch['values'], returns, epochs=args.epochs, batch_size=args.critic_batch_size,
                                    partial_values=batch['partial_values'])
        value_stats['predictionByPlayLength'] = prediction_metrics(batch['values'], returns, batch['play_lengths'])
        return actor_stats, value_stats, time.monotonic() - started

    # --------------------------------------------------------------- league
    def snapshot_league(self, stats):
        pair = [copy.deepcopy(m).eval().requires_grad_(False) for m in self.models]
        self.histories.append(pair)
        self.matchmaker.append()
        self.frozen_before.append([copy.deepcopy(m.state_dict()) for m in pair])
        self.descriptors.append((stats['behavior'] / max(1, stats['behavior_count'])).tolist())

    def prune_league(self):
        """Keep the newest eight plus any opponent still serving an episode, or the diverse archive."""
        if self.args.variant in RECENT_ONLY_VARIANTS:
            newest = range(max(0, len(self.histories) - NEWEST_SNAPSHOTS), len(self.histories))
            keep = sorted(set(newest) | {int(x) for x in self.role_ids.ravel() if x >= 0})
        else:
            keep = archive_indices(self.descriptors, self.role_ids, limit=self.args.archive_limit)
        remap = {old: new for new, old in enumerate(keep)}
        for role in range(2):
            self.role_ids[:, role] = np.array([remap[int(x)] if x >= 0 else -1 for x in self.role_ids[:, role]])
        self.matchmaker.prune(keep)
        self.histories = [self.histories[i] for i in keep]
        self.frozen_before = [self.frozen_before[i] for i in keep]
        self.descriptors = [self.descriptors[i] for i in keep]

    def verify_frozen(self):
        """Frozen opponents must be bit-identical to their snapshots and carry no gradients."""
        for before, pair in zip(self.frozen_before, self.histories):
            for old, model in zip(before, pair):
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(old[name], value, atol=0, rtol=0)
                assert all(parameter.grad is None for parameter in model.parameters())

    # ----------------------------------------------------------- checkpoint
    def record(self, update, row):
        args, parent = self.args, self.parent
        rollout_state = dict(pool=self.pool.snapshot(), buttons=self.buttons.copy(), memories=[x.clone() for x in self.memories],
                             noises=self.noises.copy(), starts=self.starts.clone(), roleIds=self.role_ids.copy(),
                             burnInPrefix=[None if x is None else tuple(t.clone() for t in x) for x in self.prefixes],
                             recent=list(self.recent))
        method = getattr(args, 'training_description',
                         f'{args.encoder}, league B; capture={args.capture_credit}, value={args.value_normalization}, maps={args.map_replay}')
        model = self.models[0]
        widened = model.observation_size != LEGACY_OBSERVATION_SIZE
        return dict(
            format=TAG_FORMAT if widened else FORMAT, encoderTypes=parent['encoderTypes'], observationSize=model.observation_size,
            physicsObservationSize=model.physical_size, noiseRho=model.noise_rho, schema=SCHEMA if widened else None,
            trainingMethod=method, rolloutState=rollout_state,
            leagueEncoderTypes=[[ENTITY if isinstance(m,EntityActor) else LEGACY for m in pair] for pair in self.histories],
            leagueNoiseRho=[pair[0].noise_rho for pair in self.histories],
            curriculumState=self.progression.state_dict(),
            levelReplayState=self.level_replay.state_dict(),
            matchmakingState=self.matchmaker.state_dict(), leagueDescriptors=self.descriptors, leagueModels=[[m.state_dict() for m in pair] for pair in self.histories],
            models=[model.state_dict() for model in self.models], optimizers=[optimizer.state_dict() for optimizer in self.optimizers],
            centralCritic=self.critic.state_dict(), centralCriticOptimizer=self.critic_optimizer.state_dict(),
            centralCriticSchema=NORMALIZED_SCHEMA if isinstance(self.critic, PopArtCritic) else SCHEMA, parentDecisions=parent['decisions'],
            criticWarmupDecisions=self.provenance['criticWarmupDecisions'],
            pilotDecisions=self.interactions, totalPolicyInteractions=self.interactions,
            actorUpdateDecisions=int(self.current_counts.sum()), currentPolicyDecisions=self.current_counts.tolist(),
            historicalPolicyDecisions=self.historical_counts.tolist(), activePolicySamples=self.active_counts.tolist(),
            decisions=parent['decisions'] + self.interactions,
            decisionsDefinition='Inherited selected-pair resource lineage plus this arm total current/historical policy interactions; not a per-role count.',
            newActorUpdates=update + 1, pilotUpdates=update + 1, pilotEpisodes=self.episodes, seconds=row['seconds'],
            provenance=self.provenance, arguments=vars(args), log=self.log,
            torchRNG=torch.get_rng_state(), worldRNG=self.world_generator.bit_generator.state,
            opponentRNG=self.opponent_generator.bit_generator.state)

    def checkpoint(self, update, row):
        self.verify_frozen()
        saved = self.record(update, row)
        torch.save(saved, self.destination / 'latest.tmp')
        os.replace(self.destination / 'latest.tmp', self.destination / 'latest.pt')
        if update + 1 in self.retained:
            torch.save(saved, self.destination / f'checkpoint-{self.interactions}.pt')
        (self.destination / 'training-log.json').write_text(json.dumps(self.log, indent=2) + '\n')
        (self.destination / 'provenance.json').write_text(json.dumps(self.provenance, indent=2) + '\n')
        if self.on_checkpoint is not None:
            self.on_checkpoint(saved, self.destination)


def train(args, *, stop_requested=None, on_checkpoint=None):
    Trainer(args, stop_requested, on_checkpoint).train()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--value-normalization', choices=['none', 'popart'], default='none')
    result.add_argument('--map-replay', choices=['random', 'progress'], default='random')
    result.add_argument('--capture-credit', choices=['immediate', 'discounted-equivalent'], default='immediate')
    result.add_argument('--blocked-cost', type=float, default=0.)
    result.add_argument('--matchmaking', choices=['uniform', 'seeker-curriculum'], default='uniform')
    result.add_argument('--noise-rho', type=float, default=0., help='AR(1) correlation of the observed exploration noise')
    result.add_argument('--curriculum', choices=['none', 'staged'], default='none', help='Arena stages gated on the seeker find rate')
    result.add_argument('--variant', choices=['full', 'baseline', 'short', 'unbalanced', 'recent-only', 'short-memory', 'slow-seeker'],
                        default='full')
    for argument in ['parent', 'critic', 'initial', 'output', 'protocol']:
        result.add_argument('--' + argument, required=True)
    result.add_argument('--history', nargs='+', required=True)
    result.add_argument('--encoder', choices=['legacy', 'entity'], required=True)
    result.add_argument('--arm', choices=['B'], default='B')
    result.add_argument('--resume')
    result.add_argument('--envs', type=int, default=128)
    result.add_argument('--workers', type=int, default=8)
    result.add_argument('--horizon', type=int, default=256)
    result.add_argument('--sequence-length', type=int, default=32)
    result.add_argument('--burn-in', type=int, default=0)
    result.add_argument('--snapshot-every', type=int, default=16)
    result.add_argument('--archive-limit', type=int, default=24)
    result.add_argument('--sequence-batch', type=int, default=256)
    result.add_argument('--critic-batch-size', type=int, default=2048)
    result.add_argument('--epochs', type=int, default=2)
    result.add_argument('--learning-rate', type=float, default=.0001)
    result.add_argument('--critic-learning-rate', type=float, default=.0001)
    result.add_argument('--entropy', type=float, default=.005)
    result.add_argument('--kl-limit', type=float, default=.008)
    result.add_argument('--target-interactions', type=int, default=15990784)
    result.add_argument('--stop-after-updates', type=int)
    result.add_argument('--retain-updates', default='122,244')
    result.add_argument('--save-every', type=int, default=8)
    result.add_argument('--seed', type=int, default=885713)
    return result


def main(args):
    stopped = False

    def stop(signum, _frame):
        nonlocal stopped
        stopped = True
        print(json.dumps(dict(stopRequested=signal.Signals(signum).name,
                              behavior='Finish this rollout/update, atomically save optimizer/RNG state, then stop.')), flush=True)

    previous = {sig: signal.getsignal(sig) for sig in [signal.SIGINT, signal.SIGTERM]}
    for sig in previous:
        signal.signal(sig, stop)
    args.graceful_worker_signals = True
    try:
        train(args, stop_requested=lambda: stopped)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main(parser().parse_args())
