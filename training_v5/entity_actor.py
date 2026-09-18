"""Original object-set encoder, isolated from the accepted browser policies.

Only the encoder changes. Sensors, recurrent state, action distributions and
Keep/Press/Release semantics are inherited from our own persistent actor.
"""
import copy
import math

import torch
from torch import nn
from persistent_actor import PersistentActor, FORMAT as LEGACY_FORMAT, OBSERVATIONS as LEGACY_SIZE

FORMAT = 'original-mujoco-relational-jump-policy-pair-v4'
TAG_FORMAT = 'original-mujoco-relational-tag-rounds-pair-v6'
ENTITY = 'object-relations-jump-v4'
LEGACY = 'legacy-linear-v1'
OBJECT_COLUMNS = slice(18, 178)   # ten object slots of sixteen values
FIXED_INDICES = list(range(18)) + list(range(178, 210))


def fixed_indices(observation_size):
    """Everything except the ten object slots: self and opponent, rays, extras, buttons, noise."""
    return list(range(18)) + list(range(178, int(observation_size)))


class ObjectSetEncoder(nn.Module):
    def __init__(self, encoder_size=96, embedding_size=32, observation_size=210):
        super().__init__()
        self.encoder_size, self.embedding_size = encoder_size, embedding_size
        self.observation_size = int(observation_size)
        self.fixed_indices = fixed_indices(self.observation_size)
        self.fixed = nn.Linear(len(self.fixed_indices), encoder_size)
        self.object_linear = nn.Parameter(torch.empty(encoder_size, 16))
        self.embedding = nn.Sequential(nn.Linear(16, embedding_size), nn.Tanh(),
                                       nn.Linear(embedding_size, embedding_size), nn.Tanh())
        self.relations = nn.MultiheadAttention(embedding_size, 4, batch_first=True)
        self.relation_scale = nn.Parameter(torch.tensor(0.0))
        self.query = nn.Linear(len(self.fixed_indices), embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)
        self.residual = nn.Linear(embedding_size, encoder_size)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.object_linear, math.sqrt(2))
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def features(self, observation):
        if observation.shape[-1] != self.observation_size:
            raise ValueError(f'This actor requires exactly {self.observation_size} observations')
        fixed = observation[..., self.fixed_indices].clone()
        # Extra defensive masking preserves every valid native observation.
        fixed[..., 11:18] = torch.where(fixed[..., 10:11] > .5,
                                       fixed[..., 11:18], 0)
        objects = observation[..., 18:178].reshape(*observation.shape[:-1], 10, 16)
        visible = objects[..., 0] > .5
        objects = torch.where(visible[..., None], objects, 0)
        return fixed, objects, visible

    def forward(self, observation):
        fixed, objects, visible = self.features(observation)
        linear = nn.functional.linear(objects, self.object_linear).sum(-2)
        embedded = self.embedding(objects)
        shape = embedded.shape
        tokens = embedded.reshape(-1, 10, self.embedding_size)
        mask = visible.reshape(-1, 10)
        tokens = torch.cat((tokens, torch.zeros_like(tokens[:, :1])), dim=1)
        padding = torch.cat((~mask, torch.zeros_like(mask[:, :1])), dim=1)
        related, _ = self.relations(tokens, tokens, tokens, key_padding_mask=padding, need_weights=False)
        embedded = embedded + self.relation_scale * related[:, :10].reshape(shape)
        keys = self.key(embedded)
        values = torch.where(visible[..., None], self.value(embedded), 0)
        logits = (keys * self.query(fixed)[..., None, :]).sum(-1) / math.sqrt(self.embedding_size)
        logits = logits.masked_fill(~visible, -torch.inf)
        # A constant zero-logit, zero-value null token keeps empty sets finite.
        logits = torch.cat((logits, torch.zeros_like(logits[..., :1])), -1)
        weights = torch.softmax(logits, -1)[..., :10]
        pooled = (weights[..., None] * values).sum(-2)
        # Mask the entire residual on empty sets, including its learned bias.
        residual = torch.where(visible.any(-1, keepdim=True), self.residual(pooled), 0)
        return self.fixed(fixed) + linear + residual

    def project(self, original):
        if original['encoder.weight'].shape != (96, 210):
            raise ValueError('Projection requires our original persistent encoder')
        with torch.no_grad():
            self.fixed.weight.copy_(original['encoder.weight'][:, FIXED_INDICES])
            self.fixed.bias.copy_(original['encoder.bias'])
            self.object_linear.copy_(original['encoder.weight'][:, 18:178].reshape(96, 10, 16).mean(1))
            self.residual.weight.zero_()
            self.residual.bias.zero_()


class EntityActor(PersistentActor):
    def __init__(self, hidden_size=64, encoder_size=96, embedding_size=32, observation_size=210, noise_rho=0.):
        super().__init__(hidden_size=hidden_size, encoder_size=encoder_size, observation_size=observation_size, noise_rho=noise_rho)
        self.encoder = ObjectSetEncoder(encoder_size, embedding_size, observation_size)

    def project(self, original):
        compatible = {key: value for key, value in original.items()
                      if not key.startswith('encoder.')}
        result = self.load_state_dict(compatible, strict=False)
        if result.unexpected_keys or any(not key.startswith('encoder.') for key in result.missing_keys):
            raise ValueError('Projection must preserve every compatible GRU/head parameter')
        self.encoder.project(original)
        return self


def build_actor(kind, hidden_size, encoder_size, embedding_size=None, observation_size=210, noise_rho=0.):
    if kind == ENTITY:
        return EntityActor(hidden_size, encoder_size, embedding_size, observation_size, noise_rho)
    if kind == LEGACY:
        return PersistentActor(hidden_size, encoder_size, observation_size, noise_rho)
    raise ValueError('Unknown encoder type')


def inferred_observation_size(state, kind):
    """The input width an actor's saved weights were trained on."""
    if kind == ENTITY:
        return int(state['encoder.fixed.weight'].shape[1]) - 18 + 178
    return int(state['encoder.weight'].shape[1])


def actor_noise(record, observation_size):
    """Original 210-input actors never observe noise; wider actors use the record's correlation."""
    if observation_size == LEGACY_SIZE:
        return 0.
    return float(record.get('noiseRho', 0.))


def load_pair(path_or_record, frozen=True):
    record = (torch.load(path_or_record, map_location='cpu', weights_only=False)
              if not isinstance(path_or_record, dict) else path_or_record)
    if record['format'] not in [FORMAT, TAG_FORMAT, LEGACY_FORMAT]:
        raise ValueError('Only our persistent or explicitly typed entity pair is supported')
    types = record.get('encoderTypes') if record['format'] != LEGACY_FORMAT else [LEGACY, LEGACY]
    if not isinstance(types, list) or len(record.get('models', [])) != 2:
        raise ValueError('A typed pair requires two model records and explicit encoder types')
    if len(types) != 2 or any(kind not in [ENTITY, LEGACY] for kind in types):
        raise ValueError('Two explicit encoder types are required')
    models = []
    for kind, state in zip(types, record['models']):
        hidden = state['memory.weight_hh'].shape[1]
        encoder = state['memory.weight_ih'].shape[1]
        embedding = state['encoder.embedding.0.weight'].shape[0] if kind == ENTITY else None
        size = inferred_observation_size(state, kind)
        models.append(build_actor(kind, hidden, encoder, embedding, size, actor_noise(record, size)))
    for model, state in zip(models, record['models']):
        model.load_state_dict(state)
        if frozen:
            model.eval().requires_grad_(False)
    return models, record


def export_actor(model):
    return dict(encoderType=ENTITY if isinstance(model, EntityActor) else LEGACY,
        observationSize=model.observation_size, physicsObservationSize=model.physical_size, noiseRho=model.noise_rho,
        actionSize=6, movementSize=4, hiddenSize=model.hidden_size, encoderSize=model.encoder_size,
        embeddingSize=model.encoder.embedding_size if isinstance(model, EntityActor) else None,
        weights={name: value.detach().cpu().float().tolist()
                 for name, value in model.state_dict().items() if not name.startswith('value.')})


def prepare_pair(parent, *, projected, seed=785611, observation_size=LEGACY_SIZE, noise_rho=0.):
    """Project our own frozen weights, reset both arms' actor optimizers alike.

    A wider ``observation_size`` produces the tag-round schema: the projected or
    copied actors are widened with zero weights on the new inputs.
    """
    kind = ENTITY if projected else LEGACY
    widened = int(observation_size) != LEGACY_SIZE
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        models = [EntityActor().project(state) if projected else PersistentActor()
                  for state in parent['models']]
        if not projected:
            for model, state in zip(models, parent['models']):
                model.load_state_dict(state)
        if widened:
            from widen_schema import widen_models
            models = widen_models(models, [kind, kind], observation_size, noise_rho)
        optimizers = [torch.optim.Adam(model.parameters(), lr=.0001, eps=1e-5)
                      for model in models]
    result = dict(format=TAG_FORMAT if widened else FORMAT, encoderTypes=[kind] * 2,
        observationSize=models[0].observation_size, physicsObservationSize=models[0].physical_size,
        noiseRho=models[0].noise_rho, schema=tag_schema() if widened else None,
        models=[copy.deepcopy(model.state_dict()) for model in models],
        optimizers=[optimizer.state_dict() for optimizer in optimizers],
        torchRNG=parent['torchRNG'].clone(), decisions=parent['decisions'],
        parentDecisions=parent['decisions'], pilotDecisions=0, newActorUpdates=0,
        provenance=copy.deepcopy(parent['provenance']))
    result['provenance']['encoderPreparation'] = dict(
        projected=projected, seed=seed, optimizerReset=True, observationSize=models[0].observation_size,
        noiseRho=models[0].noise_rho,
        description='Own six slot kernels averaged; compatible weights retained; attention residual starts at zero.'
                    if projected else 'Exact selected legacy weights; matched fresh Adam state.',
        inheritedCountDefinition=parent.get('decisionsDefinition'))
    return result


def tag_schema():
    from game import SCHEMA
    return SCHEMA


def restore_league(record):
    """Restore each frozen pair's encoder instead of assuming the current type.

    Earlier full-state saves omitted league types; their native state keys
    distinguish the two supported encoder families unambiguously. Pairs on the
    original schema never observe noise; wider pairs use the saved correlation.
    """
    pairs = record['leagueModels']
    kinds = record.get('leagueEncoderTypes')
    if kinds is None:
        kinds = []
        for pair in pairs:
            types = []
            for state in pair:
                if 'encoder.embedding.0.weight' in state:
                    types.append(ENTITY)
                elif 'encoder.weight' in state:
                    types.append(LEGACY)
                else:
                    raise ValueError('Unknown saved league encoder')
            kinds.append(types)
    if len(kinds) != len(pairs):
        raise ValueError('Missing saved league encoder identities')
    noises = record.get('leagueNoiseRho') or [record.get('noiseRho', 0.)] * len(pairs)
    if len(noises) != len(pairs):
        raise ValueError('Missing saved league noise correlations')
    return [load_pair(dict(format=TAG_FORMAT, models=pair, encoderTypes=types, noiseRho=rho))[0]
            for pair, types, rho in zip(pairs, kinds, noises)]
