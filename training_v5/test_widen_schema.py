"""Widening to the v6 schema keeps every output bit-for-bit until the new weights move."""
import unittest

import torch

from entity_actor import ENTITY, LEGACY, TAG_FORMAT, EntityActor, load_pair, prepare_pair
from game import NOISE, PHYSICAL_OBSERVATIONS, SCHEMA
from persistent_actor import PersistentActor
from widen_schema import physical_size, source_column, widen_models, widen_record

torch.set_num_threads(1)
WIDE = PHYSICAL_OBSERVATIONS + 2 + NOISE


def embed(old_rows, physical, noise_rho):
    """Place 210-column rows into the wider layout with random extras and noise."""
    count = old_rows.shape[0]
    extras = torch.randn(count, physical - 208)
    noise = torch.randn(count, NOISE) if noise_rho else torch.zeros(count, 0)
    return torch.cat([old_rows[:, :208], extras, old_rows[:, 208:210], noise], dim=1)


class WidenSchemaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1201)

    def test_column_mapping(self):
        self.assertEqual(physical_size(WIDE, .7), PHYSICAL_OBSERVATIONS)
        self.assertEqual(physical_size(216, 0), PHYSICAL_OBSERVATIONS)
        self.assertEqual([source_column(i, 214) for i in (0, 207, 208, 213, 214, 215, 216, 219)],
                         [0, 207, None, None, 208, 209, None, None])

    def test_widened_actors_reproduce_the_original_outputs_exactly(self):
        for kind, model in ((ENTITY, EntityActor()), (LEGACY, PersistentActor())):
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(torch.randn_like(parameter) * .05)
            for noise_rho in (0., .7):
                wide = widen_models([model], [kind], WIDE if noise_rho else 216, noise_rho)[0]
                self.assertEqual(wide.physical_size, PHYSICAL_OBSERVATIONS)
                rows = torch.randn(16, 210) * .4
                rows[:, 5] = 1
                memory = torch.randn(16, 64)
                with torch.no_grad():
                    old = model(rows, memory)
                    new = wide(embed(rows, PHYSICAL_OBSERVATIONS, noise_rho), memory)
                torch.testing.assert_close(new[2], old[2], atol=0, rtol=0)
                torch.testing.assert_close(new[3], old[3], atol=0, rtol=0)
                torch.testing.assert_close(new[1].logits, old[1].logits, atol=0, rtol=0)
                if noise_rho:
                    # The returned mean carries the observed noise; the head itself is unchanged.
                    observed = embed(rows, PHYSICAL_OBSERVATIONS, noise_rho)
                    with torch.no_grad():
                        carried = wide(observed, memory)[0].mean - noise_rho * observed[:, -NOISE:]
                    torch.testing.assert_close(carried, old[0].mean, atol=1e-6, rtol=1e-6)
                else:
                    torch.testing.assert_close(new[0].mean, old[0].mean, atol=0, rtol=0)

    def test_record_widening_and_prepared_pairs_carry_the_schema(self):
        source = PersistentActor()
        parent = dict(models=[source.state_dict(), source.state_dict()], torchRNG=torch.get_rng_state(), decisions=5, provenance={})
        prepared = prepare_pair(parent, projected=True, observation_size=WIDE, noise_rho=.7)
        self.assertEqual(prepared['format'], TAG_FORMAT)
        self.assertEqual((prepared['observationSize'], prepared['physicsObservationSize'], prepared['noiseRho']), (WIDE, 214, .7))
        self.assertEqual(prepared['schema'], SCHEMA)
        models, _ = load_pair(prepared)
        self.assertEqual([m.observation_size for m in models], [WIDE, WIDE])
        self.assertEqual([m.noise_rho for m in models], [.7, .7])
        narrow = prepare_pair(parent, projected=True)
        self.assertEqual(narrow['observationSize'], 210)
        self.assertEqual(narrow['noiseRho'], 0.)
        record = dict(narrow, format='original-mujoco-relational-jump-policy-pair-v4', torchRNG=torch.get_rng_state(), provenance={})
        widened = widen_record(record, WIDE, .7)
        self.assertEqual(widened['encoderTypes'], [ENTITY, ENTITY])
        self.assertTrue(all(not state['state'] for state in widened['optimizers']))
        self.assertEqual(widened['provenance']['schemaMigration']['toObservationSize'], WIDE)
        for original, wide in zip(load_pair(record)[0], load_pair(widened)[0]):
            rows = torch.randn(8, 210) * .3
            memory = torch.zeros(8, 64)
            with torch.no_grad():
                torch.testing.assert_close(wide(embed(rows, 214, .7), memory)[3], original(rows, memory)[3], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            widen_record(widened, WIDE, .7)


if __name__ == '__main__':
    unittest.main()
