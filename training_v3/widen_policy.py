"""Function-preserving capacity expansion of our own recurrent entity policies.

Existing units retain their complete recurrence. Extra units initially have no
influence on old units or action heads, but their connections are trainable.
Optimizer moments are explicitly reset; this is a migration, not exact resume.
"""
import math
import torch
from entity_actor import EntityActor
from residual_critic import ResidualCentralCritic


def widen_actor(source, hidden_size=256, encoder_size=256, embedding_size=128):
    if not isinstance(source, EntityActor):
        raise ValueError('Only the original entity actor is supported')
    h, e, d = source.hidden_size, source.encoder_size, source.encoder.embedding_size
    if hidden_size < h or encoder_size < e or embedding_size < d:
        raise ValueError('Widening cannot discard trained units')
    target = EntityActor(hidden_size, encoder_size, embedding_size)
    original, result = source.state_dict(), target.state_dict()
    with torch.no_grad():
        # Preserve the original object embeddings as an independent subspace.
        for key in ['encoder.fixed.weight', 'encoder.fixed.bias', 'encoder.object_linear',
                    'encoder.embedding.0.weight', 'encoder.embedding.0.bias']:
            slices = tuple(slice(0, n) for n in original[key].shape)
            result[key][slices].copy_(original[key])
        for base in ['encoder.embedding.2', 'encoder.key', 'encoder.value', 'encoder.residual']:
            old = original[base + '.weight']; rows, cols = old.shape
            result[base + '.weight'][:rows].zero_()
            result[base + '.weight'][:rows, :cols].copy_(old)
            result[base + '.bias'][:rows].copy_(original[base + '.bias'])
        # Extra attention keys have zero matching query components. Compensate
        # for the new sqrt(d) normalization to preserve attention probabilities.
        factor = math.sqrt(embedding_size / d)
        result['encoder.query.weight'].zero_(); result['encoder.query.bias'].zero_()
        result['encoder.query.weight'][:d].copy_(original['encoder.query.weight'] * factor)
        result['encoder.query.bias'][:d].copy_(original['encoder.query.bias'] * factor)
        for gate in range(3):
            old_rows = slice(gate*h, (gate+1)*h)
            new_rows = slice(gate*hidden_size, gate*hidden_size+h)
            for name, columns in [('weight_ih', e), ('weight_hh', h)]:
                result['memory.'+name][new_rows].zero_()
                result['memory.'+name][new_rows, :columns].copy_(original['memory.'+name][old_rows])
            for name in ['bias_ih', 'bias_hh']:
                result['memory.'+name][new_rows].copy_(original['memory.'+name][old_rows])
        for head in ['movement', 'tools', 'value']:
            result[head+'.weight'].zero_()
            result[head+'.weight'][:, :h].copy_(original[head+'.weight'])
            result[head+'.bias'].copy_(original[head+'.bias'])
        result['log_std'].copy_(original['log_std'])
        target.load_state_dict(result)
    return target


def widen_critic(source_state, hidden_size=256):
    key = 'correction.value.0.weight'
    old_h = (source_state[key].shape[1] - 228) // 2
    if hidden_size < old_h: raise ValueError('Cannot shrink critic memory inputs')
    target = ResidualCentralCritic(.1, memory_size=hidden_size)
    result = target.state_dict()
    with torch.no_grad():
        for name, value in source_state.items():
            if name == key:
                result[name].zero_()
                result[name][:, :228].copy_(value[:, :228])
                for role in range(2):
                    result[name][:, 228+role*hidden_size:228+role*hidden_size+old_h].copy_(
                        value[:, 228+role*old_h:228+(role+1)*old_h])
            else: result[name].copy_(value)
        target.load_state_dict(result)
    return target
