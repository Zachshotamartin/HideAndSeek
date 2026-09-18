import unittest
import numpy as np
import torch
from torch.distributions import Independent, Normal, Categorical, TransformedDistribution, TanhTransform
from actor import PhysicalActor
from persistent_actor import PersistentActor, advance_buttons, augment

torch.set_num_threads(1)

class PersistentTests(unittest.TestCase):
    def test_uniform_head_and_exact_warm_start(self):
        torch.manual_seed(314)
        original=PhysicalActor(138);pilot=PersistentActor().warm_start(original.state_dict())
        physical=torch.randn(5,138);memory=torch.randn(5,64);obs=torch.cat((physical,torch.zeros(5,2)),-1)
        with torch.no_grad():
            old=original(physical,memory);new=pilot(obs,memory)
        torch.testing.assert_close(old[0].mean,new[0].mean,rtol=0,atol=0)
        torch.testing.assert_close(old[2],new[2],rtol=0,atol=0)
        torch.testing.assert_close(old[3],new[3],rtol=0,atol=0)
        torch.testing.assert_close(new[1].probs,torch.full((5,2,3),1/3))
        self.assertEqual(torch.count_nonzero(pilot.encoder.weight[:,138:]).item(),0)

    def test_keep_press_release_have_no_minimum_hold_and_prep_does_not_latch(self):
        state=np.zeros((2,2),np.float32)
        state=advance_buttons(state,[[1,0],[1,1]],[False,True])
        np.testing.assert_array_equal(state,[[1,0],[0,0]])
        state=advance_buttons(state,[[0,1],[0,0]])
        np.testing.assert_array_equal(state,[[1,1],[0,0]])
        state=advance_buttons(state,[[2,2],[1,2]])
        np.testing.assert_array_equal(state,[[0,0],[1,0]])
        state=advance_buttons(state,[[0,0],[2,0]])
        np.testing.assert_array_equal(state,np.zeros((2,2)))
        with self.assertRaises(ValueError):advance_buttons([0,0],[3,0])

    def test_actor_critic_see_only_own_button_states_and_reset_is_zero(self):
        physical=np.zeros((3,2,138),np.float32);buttons=np.zeros((3,2,2),np.float32)
        buttons[1,0]=[1,0];obs=augment(physical,buttons)
        self.assertEqual(obs.shape,(3,2,140))
        np.testing.assert_array_equal(obs[1,0,-2:],[1,0]);np.testing.assert_array_equal(obs[1,1,-2:],[0,0])
        buttons[1]=0;np.testing.assert_array_equal(augment(physical,buttons)[1,:,-2:],np.zeros((2,2)))

    def test_categorical_command_likelihood_matches_executed_command_density(self):
        movement=torch.tensor([[.2,-.6,1.1]],requires_grad=True)
        normal=Normal(torch.zeros(1,3),torch.ones(1,3)*.7)
        categorical=Categorical(logits=torch.tensor([[[.1,.2,.7],[-.3,.4,.2]]]))
        commands=torch.tensor([[1.,0.]])
        actual,_=PersistentActor.statistics(normal,categorical,torch.cat((movement,commands),-1),False)
        expected=Independent(TransformedDistribution(normal,[TanhTransform(cache_size=1)]),1).log_prob(torch.tanh(movement))+categorical.log_prob(commands.long()).sum(-1)
        torch.testing.assert_close(actual,expected)
        self.assertNotEqual(float(categorical.log_prob(commands.long()).sum()),float(categorical.log_prob(torch.tensor([[0,0]])).sum()))

    def test_deterministic_decisions_do_not_consume_rng(self):
        model=PersistentActor();obs=torch.zeros(1,140);memory=torch.zeros(1,64)
        before=torch.get_rng_state().clone();out=model.act(obs,memory,True)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(out[1],torch.zeros(1,2,dtype=torch.int64)))

    def test_entropy_restores_saturated_movement_without_changing_tool_reward(self):
        mean=torch.full((256,3),12.,requires_grad=True)
        logits=torch.tensor([[[0.,0.,0.],[0.,0.,0.]]]).repeat(256,1,1).requires_grad_()
        normal=Normal(mean,torch.ones_like(mean)*.2)
        tools=Categorical(logits=logits)
        _,entropy=PersistentActor.statistics(normal,tools,torch.cat((mean,torch.zeros(256,2)),-1))
        self.assertTrue(torch.isfinite(entropy).all())
        entropy.mean().backward()
        self.assertTrue((mean.grad<0).all())
        torch.testing.assert_close(logits.grad,torch.zeros_like(logits.grad),atol=1e-7,rtol=0)

    def test_extreme_means_and_categorical_logits_keep_finite_gradients(self):
        mean=torch.tensor([[100.,-100.,0.]],requires_grad=True)
        logits=torch.tensor([[[100.,-100.,0.],[-100.,100.,0.]]],requires_grad=True)
        normal=Normal(mean,torch.ones_like(mean)*.1);tools=Categorical(logits=logits)
        raw=torch.cat((mean.detach(),torch.tensor([[0.,1.]])),-1)
        logp,entropy=PersistentActor.statistics(normal,tools,raw)
        (logp+entropy).sum().backward()
        self.assertTrue(torch.isfinite(logp).all() and torch.isfinite(entropy).all())
        self.assertTrue(torch.isfinite(mean.grad).all() and torch.isfinite(logits.grad).all())

if __name__=='__main__':unittest.main()
