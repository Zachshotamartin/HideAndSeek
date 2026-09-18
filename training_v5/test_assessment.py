"""Current six-action policies, pursuit fixtures, seed uncertainty and fresh initialization."""
import argparse
import tempfile
import json
from unittest.mock import patch
import unittest
from pathlib import Path
import numpy as np
import torch
from assess_behavior import assess
from behavior_fixtures import FIXTURES,fixture_arena,target_action
from comparison_summary import summarize_comparison
from compare_curriculum import prepare_trial,arms_for
from entity_actor import load_pair
from persistent_evaluate import sample,episode
from physics import PhysicsEnv
from synthetic import synthetic_source,synthetic_cohort


class AssessmentTests(unittest.TestCase):
    def test_fixture_sensors_match_intended_start_and_target_is_physical(self):
        for fixture in FIXTURES:
            e=PhysicsEnv(arena=fixture_arena(fixture,1900100100),prep=0,play=60)
            try:
                self.assertEqual(bool(e.seen[1,0]),fixture!='blocked-path')
                self.assertEqual(e.n_obj,0)
                self.assertEqual(target_action(fixture,e).shape,(6,))
                if fixture=='reacquisition':
                    original=e.data.qpos.copy()
                    for _ in range(30):
                        actions=np.zeros((2,6));actions[0]=target_action(fixture,e)
                        e.step(actions)
                    self.assertGreater(e.data.qpos[1],original[1]+.3)
                    self.assertFalse(e.seen[1,0])
            finally:e.close()

    def test_mean_actions_include_jump_buttons_and_native_modes_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            path=synthetic_source(Path(folder)/'pair.pt')
            models=load_pair(path)[0]
            e=PhysicsEnv(arena=fixture_arena('visible-target',1900100100),prep=0,play=8)
            try:
                view=e.observe()[1];mem=torch.zeros(1,models[1].hidden_size)
                with torch.no_grad():
                    a=sample(models[1],view,mem,np.zeros(2),np.random.default_rng(1),True)
                    b=sample(models[1],view,mem,np.zeros(2),np.random.default_rng(99),True)
                np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[0].shape,(6,))
                for modes in [('mean','mean'),('sample','sample')]:
                    a=episode(models,1900100100,'open','learned',arena_config=dict(prep=0,play=8),role_modes=modes,fixture='visible-target')
                    b=episode(models,1900100100,'open','learned',arena_config=dict(prep=0,play=8),role_modes=modes,fixture='visible-target')
                    self.assertEqual(a,b)
                report=assess(path,[1900100100],play=8)
                self.assertEqual(len(report['episodes']),10)
                self.assertFalse(report['automaticPublication'])
            finally:e.close()

    def test_fresh_actors_matched_critic_league_rng_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=synthetic_source(root/'source.pt');heldout=synthetic_source(root/'heldout.pt',seed=41)
            cohort=synthetic_cohort(root/'cohort.json')
            args=argparse.Namespace(source=str(source),heldout=str(heldout),cohort=str(cohort),history=[],steps=65536,suite='initialization')
            configs={}
            for arm,settings in arms_for('initialization'):
                dest=root/arm;dest.mkdir()
                configs[arm]=prepare_trial(args,dest,42,arm,settings)
                (dest/'run').mkdir();(dest/'run/latest.pt').touch()
                again=prepare_trial(args,dest,42,arm,settings)
                self.assertEqual(configs[arm],again)
            saved={k:torch.load(c['parent'],weights_only=False) for k,c in configs.items()}
            critics={k:torch.load(c['critic'],weights_only=False) for k,c in configs.items()}
            for key in critics['fresh']['centralCritic']:
                torch.testing.assert_close(critics['fresh']['centralCritic'][key],critics['continued']['centralCritic'][key],rtol=0,atol=0)
            torch.testing.assert_close(saved['fresh']['torchRNG'],saved['continued']['torchRNG'],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(saved['fresh']['models'][0][k],saved['continued']['models'][0][k]) for k in saved['fresh']['models'][0]))
            from persistent_train import file_hash
            self.assertEqual([file_hash(p) for p in configs['fresh']['history']], [file_hash(p) for p in configs['continued']['history']])

    def test_both_initialization_arms_complete_native_updates(self):
        from train_entity import train
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=synthetic_source(root/'source.pt');heldout=synthetic_source(root/'heldout.pt',seed=41)
            cohort=synthetic_cohort(root/'cohort.json')
            args=argparse.Namespace(source=str(source),heldout=str(heldout),cohort=str(cohort),history=[],steps=65536,suite='initialization')
            def world(*a,**k): return dict(scenario='open',size=8,n_boxes=1,n_ramps=0,prep=0,play=8)
            for arm,settings in arms_for('initialization'):
                dest=root/arm;dest.mkdir()
                config=prepare_trial(args,dest,42,arm,settings)
                config.update(envs=4,workers=1,horizon=16,sequence_length=8,burn_in=0,sequence_batch=4,
                              target_interactions=128,retain_updates='',epochs=1,critic_batch_size=64)
                protocol=Path(config['protocol']); data=json.loads(protocol.read_text())
                data['training']={k:config[k] for k in data['training']}; protocol.write_text(json.dumps(data))
                with patch('train_entity.environment_config',world): train(argparse.Namespace(**config))
                checkpoint=torch.load(Path(config['output'])/'latest.pt',weights_only=False)
                self.assertEqual(checkpoint['totalPolicyInteractions'],128)
                self.assertTrue(all(torch.isfinite(v).all() for model in checkpoint['models'] for v in model.values()))
                self.assertIn('Fresh central critic',checkpoint['provenance']['criticInitialization'])

    def test_final_selection_needs_independent_seeds_and_both_roles(self):
        rows=[dict(seed=i,metrics=dict(baseline=dict(score=.5,hiderUtility=.5,seekerUtility=.5),
            candidate=dict(score=.6,hiderUtility=.6,seekerUtility=.6))) for i in range(5)]
        self.assertTrue(summarize_comparison(rows,'baseline')['comparisons']['candidate']['eligibleForReview'])
        rows[0]['metrics']['candidate']['seekerUtility']=0
        self.assertFalse(summarize_comparison(rows,'baseline')['comparisons']['candidate']['eligibleForReview'])
        self.assertFalse(summarize_comparison(rows[:3],'baseline')['comparisons']['candidate']['eligibleForReview'])
