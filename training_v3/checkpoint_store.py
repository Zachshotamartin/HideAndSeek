"""Portable native dependencies and atomic checkpoint bookkeeping.

These helpers do not change tensors, optimizers, physics, or policy selection.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import uuid

import torch

from persistent_train import file_hash


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sync_file(path):
    with open(path, 'rb') as stream:
        os.fsync(stream.fileno())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with open(temporary, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def copy_immutable(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = file_hash(source)
    if destination.exists():
        if file_hash(destination) != digest:
            raise ValueError(f'Refusing to replace immutable artifact: {destination}')
        return digest
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    shutil.copyfile(source, temporary)
    if file_hash(temporary) != digest:
        temporary.unlink()
        raise ValueError('Source changed while snapshotting; use an immutable step checkpoint or retry')
    sync_file(temporary)
    os.replace(temporary, destination)
    return digest


def replace_from_immutable(source, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    copy_immutable(source, temporary)
    os.replace(temporary, destination)


def require_resume_state(saved):
    required = ['models', 'optimizers', 'centralCritic', 'centralCriticOptimizer',
                'torchRNG', 'worldRNG', 'opponentRNG', 'arguments', 'provenance',
                'totalPolicyInteractions', 'pilotUpdates', 'seconds']
    if any(key not in saved for key in required):
        raise ValueError('Use a full native training checkpoint, not browser JSON or a role-only export')
    if len(saved['models']) != 2 or len(saved['optimizers']) != 2:
        raise ValueError('Preserve the entire hider/seeker pair and both optimizers')


def locate_dependency(value, digest, checkpoint):
    checkpoint = Path(checkpoint).resolve()
    original = Path(value)
    roots = list(checkpoint.parents)
    candidates = [original] if original.is_absolute() else [root / original for root in roots] + [Path.cwd() / original]
    for candidate in candidates:
        if candidate.is_file() and file_hash(candidate) == digest:
            return candidate.resolve()
    # A best-model registry or copied bundle may store the exact immutable
    # dependencies by hash while preserving the original checkpoint bytes.
    for root in roots:
        for suffix in ['.pt', '.json']:
            candidate = root / 'assets' / (digest + suffix)
            if candidate.is_file() and file_hash(candidate) == digest:
                return candidate
    raise ValueError(f'Missing native dependency {digest}: {value}. Copy the whole saved run/bundle.')


def stage_dependencies(checkpoint, saved, output):
    require_resume_state(saved)
    arguments, provenance = saved['arguments'], saved['provenance']
    expected = {'parent': provenance['parentSHA256'], 'initial': provenance['initialSHA256'],
                'critic': provenance.get('criticSourceSHA256', provenance.get('criticFitSHA256'))}
    staged = {}
    for name, digest in expected.items():
        source = locate_dependency(arguments[name], digest, checkpoint)
        relative = f'assets/{digest}.pt'
        copy_immutable(source, Path(output) / relative)
        staged[name] = relative
    staged['history'] = []
    if len(arguments['history']) != len(provenance['history']):
        raise ValueError('Historical dependency metadata is incomplete')
    for value, record in zip(arguments['history'], provenance['history']):
        digest = record['sha256']
        source = locate_dependency(value, digest, checkpoint)
        relative = f'assets/{digest}.pt'
        copy_immutable(source, Path(output) / relative)
        staged['history'].append(relative)
    if 'protocol' in arguments:
        digest = provenance['protocolSHA256']
        source = locate_dependency(arguments['protocol'], digest, checkpoint)
        relative = f'assets/{digest}.json'
        copy_immutable(source, Path(output) / relative)
        staged['protocol'] = relative
    return staged


def read_native(path):
    path = Path(path)
    manifest_path = path.parent / 'MANIFEST.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('format') == 'hide-seek-native-resume-bundle-v1':
            if manifest['resumeFile'] != path.name:
                raise ValueError('Use the native resume file named by the bundle manifest')
            for relative, expected in manifest['files'].items():
                asset = (path.parent / relative).resolve()
                if not asset.is_relative_to(path.parent.resolve()) or file_hash(asset) != expected['sha256']:
                    raise ValueError('Portable native bundle integrity check failed')
    payload = path.read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    require_resume_state(saved)
    # Only unpickle one's own trusted native training files.
    return saved, hashlib.sha256(payload).hexdigest()


def load_native(path):
    return read_native(path)[0]
