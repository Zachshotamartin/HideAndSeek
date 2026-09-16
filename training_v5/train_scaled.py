"""Tag-round league training of the entity policies with saved milestone evaluations.

One native process owns the training. No strategy scripts or tool-use bonuses.
The source pair is widened to the v6 observation schema (last-seen memory,
time remaining, observed exploration noise) with zero weights on the new
inputs, so training starts from exactly the source's behaviour.
Resume with the same command; budgets can subsequently be extended using
train_saved.py, preserving complete actor/critic/optimizer checkpoints.

Every external input is explicit: the source pair, the evaluation cohort, the
frozen league anchors and the held-out evaluation opponent. Nothing is read
from paths remembered inside a checkpoint, so a fresh clone reproduces a run.
"""
import argparse
import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import torch

from capture import CAPTURE_DISTANCE
from checkpoint_store import atomic_json, copy_immutable, utc_now
from entity_actor import EntityActor, load_pair
from evaluated_models import register, utility
from game import CLOCK, MEMORY_AGE, MINIMUM_PREP, NOISE, PHYSICAL_OBSERVATIONS, PREP_FRACTION, SCHEMA
from persistent_train import file_hash
from protocol import archive_exercised, extended_cohort
from stages import MINIMUM_EPISODES, STAGES, THRESHOLD, WINDOW
from train_entity import parser as training_parser, train
from widen_schema import widen_record

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ['full', 'baseline', 'short', 'unbalanced', 'recent-only', 'short-memory', 'slow-seeker']
SNAPSHOT_EVERY = 4        # opponent snapshots per update; a 160-update pilot adds 40
ARCHIVE_LIMIT = 16        # anchors + eight newest + behaviour-diverse picks
SEQUENCE_LENGTH = {v: 256 for v in VARIANTS} | {'short-memory': 64}
BURN_IN = {v: 64 for v in VARIANTS} | {'baseline': 0, 'short-memory': 0}
NOISE_RHO = .7                    # AR(1) correlation of the observed exploration noise (0.22 s time constant at 12.5 Hz)
OBSERVATION_SIZE = PHYSICAL_OBSERVATIONS + 2 + NOISE
DEFAULT_SEED = 1091252
BATCH = 65536                     # interactions per PPO update
SNAPSHOT_INTERVAL = 1048576       # immutable full-state checkpoints about every million interactions
EVALUATION_INTERVAL = 5242880
GATE_INTERACTIONS = 10 * EVALUATION_INTERVAL   # the run must show a working seeker here or stop for review
GATE_FOUND_RATE = .5                           # candidate seeker sees the hider in at least half of the evaluation rounds
PROTOCOL_FORMAT = 'hide-seek-tag-rounds-league-v6'
PROTOCOL_FIELDS = ['variant', 'seed', 'arm', 'envs', 'workers', 'horizon', 'sequence_length', 'burn_in', 'snapshot_every',
                   'archive_limit', 'sequence_batch', 'critic_batch_size', 'epochs', 'learning_rate', 'critic_learning_rate',
                   'entropy', 'kl_limit', 'target_interactions', 'matchmaking', 'blocked_cost', 'value_normalization', 'map_replay',
                   'capture_credit', 'noise_rho', 'curriculum']
COMPARISONS = ('Unchanged physics and sensor contract. Tag rounds: a seeker within reach and in sight tags the hider and the '
               'round ends with the remaining play credited (discounted-equivalent); the hider wins by surviving the clock. '
               'Preparation grows with the play length. Actors observe last-seen opponent offset and age, the time remaining '
               'and their own coherent exploration noise; nothing tells them where an unseen opponent is. Seeker matchmaking, '
               'PopArt value normalisation, progress-based map replay and a staged arena curriculum gated on the measured '
               'seeker find rate. Current versus fixed reference, a held-out opponent outside the league, and counterfactual '
               'tools-disabled, stateless and blackout evaluations.')
TRAINING_DESCRIPTION = ('Relational attention; 10 object slots; 30 rays; visibility reward plus tag; refreshed self-play. '
                        'Balanced active samples, 15/30/60 s randomized rounds in staged arenas, diverse opponent archive, '
                        '256-step sequences with 64-step burn-in through the previous rollout; 0.1 tool entropy scale; '
                        'bounded grounded jump; 2.2 m walls; tag ends the round (v6).')


def resolve_source_history(source, source_path):
    paths = []
    for original in source.get('arguments', {}).get('history', []):
        candidate = Path(original)
        if not candidate.is_absolute():
            candidate = source_path.parent / candidate
        paths.append(candidate)
    return paths


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


# ------------------------------------------------------------- preparation
def check_existing_setup(prepared, source_path, target, seed, variant, cohort_hash):
    setup = json.loads((prepared / 'SETUP.json').read_text())
    if (setup['sourceSHA256'] != file_hash(source_path) or setup['target'] != target or setup.get('seed', DEFAULT_SEED) != seed
            or setup.get('variant', 'full') != variant or setup.get('sourceCohortSHA256') != cohort_hash):
        raise ValueError('Preserve this run configuration; branch or extend with train_saved.py')
    return setup


def widened_parent(source):
    """The source pair on the v6 schema with fresh actor optimizers; an already-widened source is copied."""
    large = load_pair(source)[0]
    if all(m.observation_size == OBSERVATION_SIZE and m.noise_rho == NOISE_RHO for m in large):
        parent = copy.deepcopy(source)
        parent.update(optimizers=[torch.optim.Adam(m.parameters()).state_dict() for m in large])
        return parent
    return widen_record(source, OBSERVATION_SIZE, NOISE_RHO)


def prepare_assets(source, source_path, prepared, seed):
    """Write the zero-update parent, the inherited critic, a reference copy and a zero-experience initial pair."""
    parent = widened_parent(source)
    large = load_pair(parent)[0]
    parent['provenance']['roleSources'] = source['provenance'].get('roleSources', source['provenance'].get('inheritedRoleSources'))
    parent.update(newActorUpdates=0)
    # Physical game is byte-identical. Only training distribution is changed.
    if parent['provenance']['physicsSHA256'] != file_hash(ROOT / 'training_v5/physics.py'):
        raise ValueError('New training requires the same v4 physical contract')
    critic_source = dict(centralCritic=source['centralCritic'], centralCriticSchema=source['centralCriticSchema'],
                         provenance=copy.deepcopy(parent['provenance']))
    torch.manual_seed(seed)
    parent['torchRNG'] = torch.get_rng_state()
    prepared.mkdir(parents=True)
    torch.save(parent, prepared / 'entity.pt')
    torch.save(critic_source, prepared / 'critic.pt')
    torch.save(parent, prepared / 'reference.pt')
    initial_record = copy.deepcopy(parent)
    initial_record.update(decisions=0, models=[
        EntityActor(m.hidden_size, m.encoder_size, m.encoder.embedding_size, OBSERVATION_SIZE, NOISE_RHO).state_dict()
        for m in large])
    torch.save(initial_record, prepared / 'initial.pt')
    return parent, large


def prepare_league(source, source_path, prepared, parent, history_paths, heldout_path):
    """Copy the frozen opponents and the held-out opponent; returns (anchor paths, held-out path or None)."""
    seen = {file_hash(prepared / 'entity.pt')}
    candidates = [Path(p) for p in history_paths] if history_paths is not None else resolve_source_history(source, source_path)
    missing = [str(c) for c in candidates if not c.exists()]
    if missing:
        raise ValueError(f'Frozen opponent checkpoints are missing; copy them or pass --history explicitly: {missing}')
    if heldout_path is None and history_paths is None and candidates:
        # The source's own earlier parent is held out of the league so that
        # evaluation has an opponent the learner never trained against.
        heldout_path, candidates = candidates[0], candidates[1:]
    histories = [str(prepared / 'entity.pt')]
    for original in candidates:
        digest = file_hash(original)
        if digest in seen:
            continue
        record = load(original)
        if record['provenance']['physicsSHA256'] != parent['provenance']['physicsSHA256']:
            raise ValueError(f'Frozen opponent {original} belongs to a different physical game')
        destination = prepared / f'history-{len(histories)}.pt'
        copy_immutable(original, destination)
        histories.append(str(destination))
        seen.add(digest)
    heldout = None
    if heldout_path is not None:
        record = load(heldout_path)
        if record['provenance']['physicsSHA256'] != parent['provenance']['physicsSHA256']:
            raise ValueError('The held-out opponent belongs to a different physical game')
        if file_hash(heldout_path) in seen:
            raise ValueError('The held-out opponent must not be a league anchor')
        copy_immutable(heldout_path, prepared / 'heldout.pt')
        heldout = str(prepared / 'heldout.pt')
    return histories, heldout


def training_arguments(prepared, output, histories, target, seed, variant):
    return training_parser().parse_args([
        '--parent', str(prepared / 'entity.pt'), '--critic', str(prepared / 'critic.pt'),
        '--initial', str(prepared / 'initial.pt'), '--history', *histories, '--output', str(output / 'run'),
        '--protocol', str(prepared / 'PROTOCOL.json'), '--encoder', 'entity',
        '--envs', '128', '--workers', '4', '--horizon', '256', '--sequence-length', str(SEQUENCE_LENGTH[variant]),
        '--burn-in', str(BURN_IN[variant]), '--snapshot-every', str(SNAPSHOT_EVERY), '--archive-limit', str(ARCHIVE_LIMIT),
        '--sequence-batch', '32', '--target-interactions', str(target),
        '--save-every', '1', '--retain-updates', '', '--seed', str(seed), '--variant', variant,
        '--matchmaking', 'seeker-curriculum', '--value-normalization', 'popart', '--map-replay', 'progress',
        '--capture-credit', 'discounted-equivalent', '--noise-rho', str(NOISE_RHO), '--curriculum', 'staged'])


def write_protocol(prepared, args, parent, large, source_path, cohort_hash, histories, heldout):
    """The predeclared comparison: settings, architecture and every asset hash."""
    architecture = [dict(hidden=m.hidden_size, encoder=m.encoder_size, embedding=m.encoder.embedding_size,
                         actorParameters=sum(p.numel() for k, p in m.named_parameters() if not k.startswith('value.')))
                    for m in large]
    protocol = dict(format=PROTOCOL_FORMAT, physicsSHA256=parent['provenance']['physicsSHA256'],
                    training={k: getattr(args, k) for k in PROTOCOL_FIELDS}, sourceCheckpointSHA256=file_hash(source_path),
                    comparisons=COMPARISONS, architecture=architecture,
                    reward='Zero-sum visibility per step; a tag (seeker within reach and in sight during play) ends the round with the '
                           'remaining play credited to the seeker at its discounted equivalent; zero preparation reward; no tool bonuses.',
                    toolEntropyScale=0.1,
                    selfPlay='70% current-current, 30% active-sample-balanced historical with seeker matchmaking; fixed anchor, '
                             'latest eight and behavioral diversity',
                    schema=dict(name=SCHEMA, observationSize=OBSERVATION_SIZE, physicalObservations=PHYSICAL_OBSERVATIONS, noise=NOISE,
                                noiseRho=NOISE_RHO, memoryAgeSeconds=MEMORY_AGE, clockSeconds=CLOCK),
                    gameRules=dict(captureDistance=CAPTURE_DISTANCE, preparationFraction=PREP_FRACTION, minimumPreparationSteps=MINIMUM_PREP),
                    curriculum=dict(stages=[dict(stage) for stage in STAGES], window=WINDOW, threshold=THRESHOLD,
                                    minimumEpisodes=MINIMUM_EPISODES),
                    gates=dict(interactions=GATE_INTERACTIONS, seekerFoundRate=GATE_FOUND_RATE,
                               rule='The candidate seeker must see the hider in at least this fraction of evaluation rounds and '
                                    'neither role may show a statistically clear regression, or the run stops for review.'),
                    evaluation=dict(sourceCohortSHA256=cohort_hash, longPlayMaps=24, heldOutOpponent=heldout is not None),
                    assets={}, history=[])
    for key, path in [('entity', args.parent), ('critic', args.critic), ('initial', args.initial)]:
        protocol['assets'][key] = dict(path=path, sha256=file_hash(path))
    for i, path in enumerate(histories):
        key = f'history-{i}'
        protocol['history'].append(key)
        protocol['assets'][key] = dict(path=path, sha256=file_hash(path))
    if heldout:
        protocol['assets']['heldout'] = dict(path=heldout, sha256=file_hash(heldout))
    atomic_json(prepared / 'PROTOCOL.json', protocol)
    return protocol


def prepare(source_path, output, target, seed=DEFAULT_SEED, variant='full', cohort_path=None, history_paths=None, heldout_path=None):
    """Freeze every input of a run under ``output/prepared`` and return its SETUP record."""
    prepared = output / 'prepared'
    if cohort_path is None:
        raise ValueError('Pass --cohort: the fixed development maps are not read from the source checkpoint')
    cohort_hash = file_hash(cohort_path)
    if prepared.exists():
        return check_existing_setup(prepared, source_path, target, seed, variant, cohort_hash)
    source = load(source_path)
    parent, large = prepare_assets(source, source_path, prepared, seed)
    histories, heldout = prepare_league(source, source_path, prepared, parent, history_paths, heldout_path)
    args = training_arguments(prepared, output, histories, target, seed, variant)
    protocol = write_protocol(prepared, args, parent, large, source_path, cohort_hash, histories, heldout)
    # Retain the old fixed maps; add construction layouts with 15/30/60 s play on fresh seeds.
    maps = json.loads(Path(cohort_path).read_text())['maps']
    atomic_json(prepared / 'cohort.json', dict(maps=extended_cohort(maps)))
    setup = dict(seed=seed, variant=variant, sourceSHA256=file_hash(source_path), target=target, arguments=vars(args),
                 createdUTC=utc_now(), architecture=protocol['architecture'], sourceCohortSHA256=cohort_hash,
                 cohortSHA256=file_hash(prepared / 'cohort.json'), anchors=histories, heldout=heldout)
    for name in ['train_scaled.py', 'protocol.py']:
        copy_immutable(ROOT / 'training_v5' / name, prepared / 'controller-source' / name)
    atomic_json(prepared / 'SETUP.json', setup)
    return setup


# -------------------------------------------------------------- controller
class Controller:
    """Owns one run folder: prepared inputs, the training process and milestone evaluations."""

    def __init__(self, options):
        self.options = options
        self.output = Path(options.output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / '.controller.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if options.stop_after_updates and not archive_exercised(options.stop_after_updates, SNAPSHOT_EVERY, 1, ARCHIVE_LIMIT):
            raise ValueError('Pilots must run long enough to exceed the opponent archive limit, otherwise the archive ablation is inert')
        setup = prepare(Path(options.source).resolve(), self.output, options.target, getattr(options, 'seed', DEFAULT_SEED),
                        getattr(options, 'variant', 'full'), cohort_path=Path(options.cohort).resolve(),
                        history_paths=getattr(options, 'history', None), heldout_path=getattr(options, 'heldout', None))
        self.args = argparse.Namespace(**setup['arguments'])
        self.args.graceful_worker_signals = True
        self.args.stop_after_updates = getattr(options, 'stop_after_updates', None)
        self.args.training_description = TRAINING_DESCRIPTION
        self.status_path = self.output / 'STATUS.json'
        if self.status_path.exists():
            self.status = json.loads(self.status_path.read_text())
        else:
            self.status = dict(createdUTC=utc_now(), evaluated=[], snapshots=[], architecture=setup['architecture'],
                               targetInteractions=options.target, publication='No automatic browser promotion')
        self.stopped = False
        self.gate_failure = None
        self.child = None
        self.heldout = self.output / 'prepared/heldout.pt'

    def save(self, **changes):
        self.status.update(changes, pid=os.getpid(), updatedUTC=utc_now())
        atomic_json(self.status_path, self.status)

    def stop(self, sig, _):
        self.stopped = True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(sig)

    def evaluate(self, path, steps):
        """Run the fixed-opponent evaluation of one immutable checkpoint and register it."""
        destination = self.output / 'evaluations' / str(steps)
        if steps in self.status['evaluated']:
            return
        if not (destination / 'evaluation.json').exists():
            command = [sys.executable, str(ROOT / 'training_v5/evaluate_saved.py'),
                       '--checkpoint', str(path), '--reference', str(self.output / 'prepared/reference.pt'),
                       '--cohort', str(self.output / 'prepared/cohort.json'), '--output', str(destination),
                       '--registry', str(self.output / 'best'), '--workers', '4']
            if self.heldout.exists():
                command += ['--heldout', str(self.heldout)]
            self.save(phase='evaluating', evaluationSteps=steps)
            with (self.output / f'evaluate-{steps}.log').open('a') as log:
                self.child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                code = self.child.wait()
                self.child = None
            if self.stopped:
                return
            if code:
                raise RuntimeError(f'Evaluation failed at {steps}; inspect evaluation log')
        # Registration is idempotent, including a restart between report and registry writes.
        register(path, self.output / 'prepared/reference.pt', destination / 'evaluation.json', self.output / 'best')
        self.status['evaluated'].append(steps)
        self.save(phase='training')
        if steps == GATE_INTERACTIONS:
            self.gate(json.loads((destination / 'evaluation.json').read_text()), steps)

    def gate(self, report, steps):
        """Stop for review unless the seeker finds the hider in most rounds and no role clearly regressed."""
        measured = utility(report)
        found = measured['seekerFoundRate']
        failures = []
        if found is None or found < GATE_FOUND_RATE:
            failures.append(f'candidate seeker found rate {found} below {GATE_FOUND_RATE}')
        failures.extend(f'statistically clear regression: {name}' for name in measured['roleRegressions'])
        record = dict(steps=steps, seekerFoundRate=found, roleRegressions=measured['roleRegressions'], passed=not failures,
                      failures=failures, checkedUTC=utc_now())
        self.status.setdefault('gates', []).append(record)
        if failures:
            self.gate_failure = record
            self.stopped = True
        self.save()

    def checkpoint(self, saved, directory):
        """Retain immutable full states at milestones and evaluate the milestone checkpoints."""
        steps = saved['totalPolicyInteractions']
        pilot_end = bool(self.args.stop_after_updates and saved['pilotUpdates'] >= self.args.stop_after_updates)
        retain = (pilot_end or steps % SNAPSHOT_INTERVAL == 0 or steps == BATCH or steps >= self.options.target or self.stopped)
        if retain:
            path = self.output / 'checkpoints' / f'{steps}.pt'
            digest = copy_immutable(directory / 'latest.pt', path)
            if steps not in self.status['snapshots']:
                self.status['snapshots'].append(steps)
            self.save(latestSnapshot=dict(steps=steps, path=str(path), sha256=digest))
        self.save(phase='training', interactions=steps, trainingSeconds=saved['seconds'],
                  currentPolicyDecisions=saved['currentPolicyDecisions'], activePolicySamples=saved['activePolicySamples'],
                  lastUpdate=saved['log'][-1])
        if not self.stopped and (pilot_end or steps == BATCH or steps % EVALUATION_INTERVAL == 0 or steps >= self.options.target):
            self.evaluate(self.output / 'checkpoints' / f'{steps}.pt', steps)

    def main(self):
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self.stop)
        try:
            latest = self.output / 'run/latest.pt'
            if latest.exists():
                self.args.resume = str(latest)
                # Re-run a pending complete milestone evaluation before further PPO.
                self.checkpoint(load(latest), latest.parent)
            if not self.stopped:
                self.save(phase='training')
                train(self.args, stop_requested=lambda: self.stopped, on_checkpoint=self.checkpoint)
            self.save(phase=self.final_phase())
        except BaseException as error:
            self.save(phase='failed', error=repr(error))
            raise


    def final_phase(self):
        if self.gate_failure:
            return 'gate-failed-awaiting-review'
        if self.stopped:
            return 'paused'
        return 'pilot-complete' if self.args.stop_after_updates else 'completed-awaiting-review'


def main(options):
    Controller(options).main()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--variant', default='full', choices=VARIANTS)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cohort', required=True, help='JSON with the fixed development maps (the earlier cohort.json)')
    p.add_argument('--history', nargs='*', default=None,
                   help='Explicit frozen opponent checkpoints; default derives them from the source and requires them to exist')
    p.add_argument('--heldout', default=None, help='Held-out evaluation opponent never placed in the league')
    p.add_argument('--stop-after-updates', type=int)
    p.add_argument('--target', type=int, default=1048576000)
    return p


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.target < BATCH or arguments.target % BATCH:
        parser().error('Use complete 65,536-interaction PPO batches')
    main(arguments)
