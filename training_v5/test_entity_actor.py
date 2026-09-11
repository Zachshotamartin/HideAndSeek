import copy
import itertools
import unittest

import numpy as np
import torch
import mujoco
from entity_actor import EntityActor, ObjectSetEncoder, prepare_pair, load_pair, ENTITY, LEGACY
from persistent_actor import PersistentActor
from league_ppo import actor_update, act_grouped
from physics import PhysicsEnv

torch.set_num_threads(1)


def observed(seed=715):
    generator = torch.Generator().manual_seed(seed)
    row = torch.randn(210, generator=generator) * .3
    row[5], row[7], row[10] = 1, 1, 1
    row[18:114].reshape(6, 16)[:, 0] = 1
    row[208:] = torch.tensor([1., 0.])
    return row


def active_attention(model):
    with torch.no_grad():
        model.encoder.residual.weight.normal_(0, .04)
        model.encoder.residual.bias.normal_(0, .03)


class EntityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(532)

    def test_all_720_permutations_with_live_attention_full_partial_empty_and_ties(self):
        model = EntityActor().eval()
        active_attention(model)
        permutations = torch.tensor(list(itertools.permutations(range(6))))
        for count in [0, 1, 3, 6]:
            row = observed(count + 12)
            objects = row[18:114].reshape(6, 16)
            objects[count:, 0] = 0
            if count == 6:
                objects[1] = objects[0]
                objects[3, 1:4] = objects[2, 1:4] + 1e-7
            batch = row.repeat(720, 1)
            batch[:, 18:114] = objects[permutations].reshape(720, 96)
            memory = torch.randn(1, 64).repeat(720, 1)
            with torch.no_grad():
                encoded = model.encoder(batch)
                normal, tools, value, next_memory = model(batch, memory)
            for result in [encoded, normal.mean, tools.logits.flatten(1), value[:, None], next_memory]:
                torch.testing.assert_close(result, result[:1].expand_as(result), atol=3e-6, rtol=3e-6)
                self.assertTrue(torch.isfinite(result).all())

    def test_shared_parameter_and_input_gradients_are_permutation_equivariant(self):
        model = EntityActor()
        active_attention(model)
        original = observed().requires_grad_()
        memory = torch.randn(1, 64)
        reference = None
        for order in list(itertools.permutations(range(6)))[::31]:
            model.zero_grad()
            row = original.detach().clone()
            row[18:114] = original.detach()[18:114].reshape(6, 16)[list(order)].flatten()
            row.requires_grad_()
            prediction = model(row[None], memory)
            loss = prediction[3].square().sum() + prediction[0].mean.sum() + prediction[2].sum()
            loss.backward()
            gradient = {name: value.grad.detach().clone() for name, value in model.named_parameters()
                        if value.grad is not None}
            restored = row.grad[18:114].reshape(6, 16)[np.argsort(order)]
            if reference is None:
                reference = (gradient, restored.clone())
            else:
                for name, value in gradient.items():
                    torch.testing.assert_close(value, reference[0][name], atol=4e-6, rtol=4e-6)
                torch.testing.assert_close(restored, reference[1], atol=3e-6, rtol=3e-6)

    def test_hidden_payloads_and_empty_set_biases_do_not_affect_output_or_gradients(self):
        model = EntityActor()
        active_attention(model)
        row = observed()
        row[10] = 0
        row[18:114].reshape(6, 16)[2:, 0] = 0
        changed = row.clone()
        changed[11:18] = 1e6
        changed[18:114].reshape(6, 16)[2:, 1:] = -1e6
        memory = torch.zeros(1, 64)
        a, b = model(row[None], memory), model(changed[None], memory)
        torch.testing.assert_close(a[3], b[3], atol=0, rtol=0)
        changed.requires_grad_()
        model(changed[None], memory)[3].sum().backward()
        self.assertEqual(changed.grad[11:18].abs().sum(), 0)
        self.assertEqual(changed.grad[18:114].reshape(6, 16)[2:].abs().sum(), 0)
        empty = observed(); empty[18:114].reshape(6, 16)[:, 0] = 0
        fixed, _, _ = model.encoder.features(empty)
        torch.testing.assert_close(model.encoder(empty), model.encoder.fixed(fixed), atol=0, rtol=0)

    def test_projection_preserves_compatible_weights_and_empty_histories_both_optimizers_reset(self):
        source = PersistentActor()
        parent = dict(models=[source.state_dict(), copy.deepcopy(source.state_dict())],
                      torchRNG=torch.get_rng_state(), decisions=123, provenance={})
        legacy, entity = [prepare_pair(parent, projected=value) for value in [False, True]]
        self.assertEqual(legacy['encoderTypes'], [LEGACY, LEGACY])
        self.assertEqual(entity['encoderTypes'], [ENTITY, ENTITY])
        self.assertTrue(all(not state['state'] for record in [legacy, entity] for state in record['optimizers']))
        self.assertTrue(torch.equal(legacy['torchRNG'], entity['torchRNG']))
        model = load_pair(entity)[0][0]
        for key, value in source.state_dict().items():
            if not key.startswith('encoder.'):
                torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
        old_memory, new_memory = torch.zeros(1, 64), torch.zeros(1, 64)
        with torch.no_grad():
            for tick in range(100):
                row = observed(tick)
                row[18:114] = 0
                old = source(row[None], old_memory)
                new = model(row[None], new_memory)
                old_memory, new_memory = old[3], new[3]
                torch.testing.assert_close(old[0].mean, new[0].mean, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(old_memory, new_memory, atol=3e-6, rtol=3e-6)

    def test_legacy_history_and_entity_current_advance_their_own_memories_and_blind_buttons(self):
        models = [EntityActor(), EntityActor()]
        histories = [[PersistentActor(), PersistentActor()]]
        physical = np.stack([np.stack([observed(i).numpy()[:208] for i in [1, 2]]) for _ in range(4)])
        physical[:, 1, 7] = 0; physical[:2, 1, 5] = .5
        memories = [torch.randn(4, 64), torch.randn(4, 64)]
        buttons = np.zeros((4, 2, 2), np.float32)
        identities = np.array([[-1, 0], [0, -1], [-1, -1], [0, 0]])
        _, next_memories, _, records = act_grouped(models, histories, physical, memories, buttons, identities, sample=False)
        for row in range(4):
            for role in range(2):
                actor = models[role] if identities[row, role] == -1 else histories[0][role]
                direct = actor(records[role][0][row:row+1], memories[role][row:row+1])[3]
                torch.testing.assert_close(next_memories[role][row], direct[0], atol=3e-6, rtol=3e-6)
        actions, _, next_buttons, _ = act_grouped(models, histories, physical, memories, buttons, identities)
        np.testing.assert_array_equal(actions[:2, 1], 0)
        np.testing.assert_array_equal(next_buttons[:2, 1], 0)

    def test_all_frozen_or_blind_entity_batches_make_no_optimizer_update(self):
        model = EntityActor()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
        obs = observed().repeat(32, 2, 1)
        obs[..., 7] = 0; obs[..., 5] = .5
        batch = (obs, torch.zeros(32, 2, 64), torch.zeros(32, 2),
                 torch.zeros(32, 2, 5), torch.zeros(32, 2))
        before = copy.deepcopy(model.state_dict())
        for role, current in [(0, torch.zeros(32, 2, dtype=torch.bool)),
                              (1, torch.ones(32, 2, dtype=torch.bool))]:
            result = actor_update(model, optimizer, batch, torch.randn(32, 2), current, role)
            self.assertEqual(result['optimizerSteps'], 0)
        self.assertFalse(optimizer.state)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_actual_hidden_opponent_and_blind_preparation_leave_actor_state_unchanged(self):
        env = PhysicsEnv(seed=8771, scenario='open', size=8, n_boxes=0, n_ramps=0)
        model = EntityActor().eval()
        active_attention(model)
        try:
            env.data.qpos[4:8] = [6, 4, .25, 0]
            for tick in [0, env.prep]:
                env.t = tick
                results = []
                for x in [1, 2]:
                    env.data.qpos[:4] = [x, 4, .25, 0]
                    mujoco.mj_forward(env.model, env.data)
                    env._senses()
                    self.assertFalse(env.seen[1, 0])
                    self.assertIsNone(env.last_seen[1])
                    row = np.concatenate((env.observe()[1], np.zeros(2, np.float32)))
                    with torch.no_grad():
                        prediction = model(torch.from_numpy(row[None]), torch.zeros(1, 64))
                    results.append((row, prediction))
                np.testing.assert_array_equal(results[0][0], results[1][0])
                for index in [0, 1, 2, 3]:
                    first = results[0][1][index]
                    second = results[1][1][index]
                    if index == 0:
                        first, second = first.mean, second.mean
                    elif index == 1:
                        first, second = first.logits, second.logits
                    torch.testing.assert_close(first, second, atol=0, rtol=0)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
