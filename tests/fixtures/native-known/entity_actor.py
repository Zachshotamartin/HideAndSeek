"""Original object-set encoder, isolated from the accepted browser policies.

Only the encoder changes. Sensors, recurrent state, action distributions and
Keep/Press/Release semantics are inherited from our own persistent actor.
"""
import copy
import math

import torch
from torch import nn
from persistent_actor import PersistentActor, FORMAT as LEGACY_FORMAT

FORMAT = 'original-mujoco-relational-jump-policy-pair-v4'
ENTITY = 'object-relations-jump-v4'
LEGACY = 'legacy-linear-v1'
FIXED_INDICES = list(range(18)) + list(range(178, 210))


class ObjectSetEncoder(nn.Module):
    def __init__(self, encoder_size=96, embedding_size=32):
        super().__init__()
        self.encoder_size, self.embedding_size = encoder_size, embedding_size
        self.fixed = nn.Linear(50, encoder_size)
        self.object_linear = nn.Parameter(torch.empty(encoder_size, 16))
        self.embedding = nn.Sequential(nn.Linear(16, embedding_size), nn.Tanh(),
                                       nn.Linear(embedding_size, embedding_size), nn.Tanh())
        self.relations = nn.MultiheadAttention(embedding_size, 4, batch_first=True)
        self.relation_scale = nn.Parameter(torch.tensor(0.0))
        self.query = nn.Linear(50, embedding_size)
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
        if observation.shape[-1] != 210:
            raise ValueError('The actor still requires exactly 210 restricted observations')
        fixed = observation[..., FIXED_INDICES].clone()
        # Position stays known; motion and facing stay visibility-gated.
        fixed[..., 14:18] = torch.where(fixed[..., 10:11] > .5,
                                       fixed[..., 14:18], 0)
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
    def __init__(self, hidden_size=64, encoder_size=96, embedding_size=32):
        super().__init__(hidden_size=hidden_size, encoder_size=encoder_size)
        self.encoder = ObjectSetEncoder(encoder_size, embedding_size)

    def project(self, original):
        compatible = {key: value for key, value in original.items()
                      if not key.startswith('encoder.')}
        result = self.load_state_dict(compatible, strict=False)
        if result.unexpected_keys or any(not key.startswith('encoder.') for key in result.missing_keys):
            raise ValueError('Projection must preserve every compatible GRU/head parameter')
        self.encoder.project(original)
        return self


def load_pair(path_or_record, frozen=True):
    record = (torch.load(path_or_record, map_location='cpu', weights_only=False)
              if not isinstance(path_or_record, dict) else path_or_record)
    if record['format'] not in [FORMAT, LEGACY_FORMAT]:
        raise ValueError('Only our persistent or explicitly typed entity pair is supported')
    types = record.get('encoderTypes') if record['format'] == FORMAT else [LEGACY, LEGACY]
    if not isinstance(types, list) or len(record.get('models', [])) != 2:
        raise ValueError('A typed pair requires two model records and explicit encoder types')
    if len(types) != 2 or any(kind not in [ENTITY, LEGACY] for kind in types):
        raise ValueError('Two explicit encoder types are required')
    models = []
    for kind, state in zip(types, record['models']):
        h = state['memory.weight_hh'].shape[1]
        e = state['memory.weight_ih'].shape[1]
        models.append(EntityActor(h, e, state['encoder.embedding.0.weight'].shape[0])
                      if kind == ENTITY else PersistentActor(h, e))
    for model, state in zip(models, record['models']):
        model.load_state_dict(state)
        if frozen:
            model.eval().requires_grad_(False)
    return models, record


def export_actor(model):
    return dict(encoderType=ENTITY if isinstance(model, EntityActor) else LEGACY,
        observationSize=210, physicsObservationSize=208, actionSize=6, movementSize=4, hiddenSize=model.hidden_size, encoderSize=model.encoder_size,
        embeddingSize=model.encoder.embedding_size if isinstance(model, EntityActor) else None,
        weights={name: value.detach().cpu().float().tolist()
                 for name, value in model.state_dict().items() if not name.startswith('value.')})


def prepare_pair(parent, *, projected, seed=785611):
    """Project our own frozen weights, reset both arms' actor optimizers alike."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        models = [EntityActor().project(state) if projected else PersistentActor()
                  for state in parent['models']]
        if not projected:
            for model, state in zip(models, parent['models']):
                model.load_state_dict(state)
        optimizers = [torch.optim.Adam(model.parameters(), lr=.0001, eps=1e-5)
                      for model in models]
    result = dict(format=FORMAT, encoderTypes=[ENTITY if projected else LEGACY] * 2,
        observationSize=210, physicsObservationSize=208,
        models=[copy.deepcopy(model.state_dict()) for model in models],
        optimizers=[optimizer.state_dict() for optimizer in optimizers],
        torchRNG=parent['torchRNG'].clone(), decisions=parent['decisions'],
        parentDecisions=parent['decisions'], pilotDecisions=0, newActorUpdates=0,
        provenance=copy.deepcopy(parent['provenance']))
    result['provenance']['encoderPreparation'] = dict(
        projected=projected, seed=seed, optimizerReset=True,
        description='Own six slot kernels averaged; compatible weights retained; attention residual starts at zero.'
                    if projected else 'Exact selected legacy weights; matched fresh Adam state.',
        inheritedCountDefinition=parent.get('decisionsDefinition'))
    return result
