import copy
import unittest
import numpy as np
import torch
from central_critic import CentralCritic, central_state, batch_states
from central_persistent_train import (advantages_and_returns, actor_update, critic_update,
    normalize_active_advantages, policy_objective, partial_value_estimates)
from fit_central_value import geometry_hash, sample_restricted
from persistent_actor import PersistentActor
from physics import PhysicsEnv

torch.set_num_threads(1)


class CentralTrainingTests(unittest.TestCase):
    def test_terminal_return_ignores_next_episode_bootstrap(self):
        rewards=torch.tensor([[.05],[-.05]])
        values=torch.tensor([[.1],[.2]]);dones=torch.tensor([[0.],[1.]])
        advantages,returns=advantages_and_returns(rewards,values,dones,torch.tensor([.3]))
        changed=advantages_and_returns(rewards,values,dones,torch.tensor([999.]))
        torch.testing.assert_close(advantages,changed[0],atol=0,rtol=0)
        torch.testing.assert_close(returns,changed[1],atol=0,rtol=0)
        self.assertAlmostEqual(float(advantages[-1]),-.25,places=6)
        self.assertAlmostEqual(float(returns[-1]),-.05,places=6)
        self.assertAlmostEqual(float(advantages[0]),.05+.998*.2-.1+.998*.98*(-.25),places=6)

    def test_discarded_seeker_preparation_has_no_adam_momentum_updates(self):
        model=PersistentActor();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        model.encoder.weight.sum().backward();optimizer.step()
        before=copy.deepcopy(model.state_dict());old_optimizer=copy.deepcopy(optimizer.state_dict())
        observed=torch.zeros(2,2,140);observed[:,:,5]=.2
        batch=[observed,torch.zeros(2,2,64),torch.zeros(2,2),torch.zeros(2,2,5),torch.zeros(2,2)]
        result=actor_update(model,optimizer,batch,torch.tensor([[.2,.1],[.4,-.2]]),role=1,
                            epochs=2,sequence_length=2,sequence_batch=2)
        self.assertEqual(result['optimizerSteps'],0)
        for name,value in model.state_dict().items():torch.testing.assert_close(value,before[name],atol=0,rtol=0)
        for key,state in optimizer.state_dict()['state'].items():
            for name,value in state.items():torch.testing.assert_close(value,old_optimizer['state'][key][name],atol=0,rtol=0)

    def test_privileged_value_update_has_no_actor_gradient_path(self):
        actor=PersistentActor();critic=CentralCritic(use_actor_memory=True)
        old_actor=copy.deepcopy(actor.state_dict());old_critic=copy.deepcopy(critic.state_dict())
        _,_,_,hidden=actor(torch.randn(2,140),torch.zeros(2,64))
        memories=torch.stack([hidden,hidden],dim=1).unsqueeze(0).repeat(2,1,1,1)
        env=PhysicsEnv(seed=7123);sample=batch_states([central_state(env),central_state(env)])
        states={key:value.unsqueeze(0).repeat(2,*([1]*value.ndim)) for key,value in sample.items()}
        with torch.no_grad():values=critic({key:value.flatten(0,1) for key,value in states.items()},memories.flatten(0,1)).reshape(2,2)
        result=critic_update(critic,torch.optim.Adam(critic.parameters(),lr=.001),states,memories,
                             values,values+.5,epochs=1,batch_size=4)
        self.assertTrue(np.isfinite(result['clippedValueMSE']))
        self.assertTrue(all(parameter.grad is None for parameter in actor.parameters()))
        for name,value in actor.state_dict().items():torch.testing.assert_close(value,old_actor[name],atol=0,rtol=0)
        self.assertTrue(any(not torch.equal(value,old_critic[name]) for name,value in critic.state_dict().items()))

    def test_map_split_hash_excludes_seed_and_agent_spawns(self):
        first=dict(seed=112,scenario='open',size=8,n_boxes=0,n_ramps=0)
        second={**first,'seed':983}
        self.assertEqual(geometry_hash(first),geometry_hash(second))
        self.assertNotEqual(geometry_hash(first),geometry_hash({**first,'n_boxes':1}))

    def test_blind_padding_does_not_dilute_objective_kl_or_normalization(self):
        def objective(padding):
            active=torch.tensor([False]*padding+[True]*3)
            advantage=torch.tensor([91.]*padding+[.4,-.2,.9])
            logp=torch.tensor([-.3]*padding+[-.2,.3,-.7],requires_grad=True)
            old=torch.tensor([-.9]*padding+[-.3,.2,-.5])
            entropy=torch.tensor([9.]*padding+[.2,.5,.7])
            normalized=normalize_active_advantages(advantage,active)
            loss,kl,count=policy_objective(logp,old,entropy,normalized,active)
            (loss/count).backward()
            return normalized[active],loss.detach()/count,kl.detach()/count,logp.grad[active]
        for padded in [1,7,29]:
            for left,right in zip(objective(0),objective(padded)):
                torch.testing.assert_close(left,right,atol=1e-7,rtol=0)

    def test_bootstrap_partial_values_match_pre_action_record_without_sampling(self):
        models=[PersistentActor(),PersistentActor()]
        envs=[PhysicsEnv(seed=632),PhysicsEnv(seed=942)]
        physical=np.stack([env.observe() for env in envs])
        memories=[torch.randn(2,64),torch.randn(2,64)]
        buttons=np.zeros((2,2,2),np.float32)
        rng=torch.get_rng_state().clone()
        with torch.no_grad():
            partial=partial_value_estimates(models,physical,memories,buttons)
        torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
        with torch.no_grad():
            _,_,_,records=sample_restricted(models,physical,memories,buttons)
        torch.testing.assert_close(partial,torch.stack([record[3] for record in records],dim=-1),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
