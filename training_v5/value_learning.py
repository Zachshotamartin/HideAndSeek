"""Optional PopArt residual critic; actor observations and game scores are unchanged."""
import torch
from residual_critic import ResidualCentralCritic, SCHEMA
from central_persistent_train import critic_update

NORMALIZED_SCHEMA = 'hide-seek-popart-residual-value-v1'
RATE = .01
MIN_STD = .1


class PopArtCritic(ResidualCentralCritic):
    def __init__(self, dropout=.1, memory_size=64):
        super().__init__(dropout, memory_size)
        self.register_buffer('return_mean', torch.zeros(()))
        self.register_buffer('return_std', torch.ones(()))

    def forward(self, state, actor_memories, partial_values):
        return self.baseline(partial_values) + self.return_std * self.residual(state, actor_memories) + self.return_mean

    @torch.no_grad()
    def update_scale(self, returns, optimizer):
        old_mean, old_std = self.return_mean.clone(), self.return_std.clone()
        mean = (1 - RATE) * old_mean + RATE * returns.mean()
        second = (1 - RATE) * (old_std.square() + old_mean.square()) + RATE * returns.square().mean()
        std = (second - mean.square()).clamp(min=MIN_STD ** 2).sqrt()
        layer = self.correction.value[-1]
        ratio = old_std / std
        layer.weight.mul_(ratio)
        layer.bias.mul_(old_std).add_(old_mean - mean).div_(std)
        for parameter in (layer.weight, layer.bias):
            state = optimizer.state.get(parameter, {})
            if 'exp_avg' in state:
                state['exp_avg'].mul_(ratio)
                state['exp_avg_sq'].mul_(ratio.square())
        self.return_mean.copy_(mean)
        self.return_std.copy_(std)


def initialize_critic(source, learning_rate, resumed=None, normalization='none'):
    if source.get('centralCriticSchema') not in (SCHEMA, NORMALIZED_SCHEMA):
        raise ValueError('Expected our residual-critic checkpoint')
    record = resumed or source
    state = record['centralCritic']
    memory_size = (state['correction.value.0.weight'].shape[1] - 228) // 2
    normalized_source = 'return_std' in state
    if resumed and normalized_source != (normalization == 'popart'):
        raise ValueError('Exact resume changed value normalization')
    critic = PopArtCritic(.1, memory_size) if normalization == 'popart' else ResidualCentralCritic(.1, memory_size)
    if normalized_source and normalization == 'none':
        # Fresh comparisons may initialize an ordinary critic from a normalized
        # source by baking the affine correction into its final residual head.
        state = {k: v.clone() for k, v in state.items()}
        state['correction.value.6.weight'].mul_(state['return_std'])
        state['correction.value.6.bias'].mul_(state['return_std']).add_(state['return_mean'])
        del state['return_mean'], state['return_std']
    result = critic.load_state_dict(state, strict=False)
    expected_missing = {'return_mean', 'return_std'} if normalization == 'popart' and not normalized_source else set()
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError('Incompatible residual critic weights')
    optimizer = torch.optim.AdamW(critic.parameters(), lr=learning_rate)
    if resumed:
        optimizer.load_state_dict(resumed['centralCriticOptimizer'])
    return critic, optimizer


def update_critic(critic, optimizer, states, memories, old_values, returns, epochs=3,
                  batch_size=512, partial_values=None):
    if not isinstance(critic, PopArtCritic):
        return critic_update(critic, optimizer, states, memories, old_values, returns,
                             epochs, batch_size, partial_values)
    critic.update_scale(returns, optimizer)
    states = {key: value.flatten(0, 1) for key, value in states.items()}
    memories = memories.flatten(0, 1).detach()
    old = ((old_values.flatten() - critic.return_mean) / critic.return_std).detach()
    targets = ((returns.flatten() - critic.return_mean) / critic.return_std).detach()
    partial_values = partial_values.flatten(0, 1).detach()
    critic.train()
    losses = []
    for _ in range(epochs):
        for selected in torch.randperm(len(targets)).split(batch_size):
            raw = critic({k: v[selected] for k, v in states.items()}, memories[selected], partial_values[selected])
            value = (raw - critic.return_mean) / critic.return_std
            clipped = old[selected] + (value - old[selected]).clamp(-.2, .2)
            loss = torch.maximum((value - targets[selected]).square(), (clipped - targets[selected]).square()).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite normalized value loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return dict(normalizedClippedValueMSE=sum(losses) / max(1, len(losses)),
                returnMean=float(critic.return_mean), returnStd=float(critic.return_std))


def prediction_metrics(values, returns, play_lengths):
    result = {}
    for play in play_lengths.unique():
        mask = play_lengths == play
        truth, prediction = returns[mask], values[mask]
        variance = truth.var(unbiased=False)
        explained = 1 - (truth - prediction).var(unbiased=False) / variance if variance > 1e-8 else None
        result[str(int(play))] = dict(samples=int(mask.sum()), mse=float((truth - prediction).square().mean()),
                                     explainedVariance=float(explained) if explained is not None else None)
    return result
