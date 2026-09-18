import unittest
import numpy as np
import torch
from entity_actor import EntityActor, load_pair, FORMAT, ENTITY
from persistent_actor import PersistentActor
from central_critic import FEATURES
from residual_critic import ResidualCentralCritic
from league_ppo import act_grouped, actor_update
from widen_policy import widen_actor, widen_critic

torch.set_num_threads(1)


class WidenTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(38251)

    def source(self):
        actor = EntityActor()
        # Test learned nonzero attention, tool and value weights, not only init.
        with torch.no_grad():
            actor.encoder.residual.weight.normal_(0, .04)
            actor.encoder.residual.bias.normal_(0, .04)
            actor.tools.weight.normal_(0, .04)
        return actor

    def test_recurrent_policy_preserved_and_extra_units_trainable(self):
        old = self.source(); new = widen_actor(old)
        a, b = torch.randn(4,64), torch.randn(4,256)
        b[:,:64] = a
        for t in range(160):
            obs = torch.randn(4,140)
            if t % 11 == 0: obs[:,18:114] = 0
            x, y, v, a = old(obs,a)
            xx, yy, vv, b = new(obs,b)
            torch.testing.assert_close(x.mean, xx.mean, atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(y.logits, yy.logits, atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(v,vv, atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(a,b[:,:64], atol=2e-6,rtol=2e-5)
            a=a.detach(); b=b.detach()
        xx.mean.square().sum().backward()
        self.assertGreater(new.movement.weight.grad[:,64:].abs().sum().item(),0)
        new.zero_grad()
        opt=torch.optim.Adam(new.parameters(),lr=.001)
        for _ in range(2):
            x,_,_,_=new(obs,b.detach());loss=(x.mean-1).square().mean()
            opt.zero_grad();loss.backward();opt.step()
        self.assertGreater(new.memory.weight_ih.grad[64:256].abs().sum().item(),0)

    def test_dimensions_round_trip(self):
        new = widen_actor(self.source())
        pair,_ = load_pair(dict(format=FORMAT,encoderTypes=[ENTITY]*2,
                              models=[new.state_dict()]*2))
        self.assertEqual(pair[1].hidden_size,256)
        self.assertEqual(pair[0].encoder.embedding_size,128)

    def test_critic_preserved_without_leaking_gradients(self):
        old=ResidualCentralCritic();old.correction.value[-1].weight.data.normal_(0,.1)
        new=widen_critic(old.state_dict());old.eval();new.eval()
        state={k:torch.randn(3,*shape) for k,shape in FEATURES.items()}
        state['objectMask'].fill_(1);state['wallMask'].fill_(1)
        a=torch.randn(3,2,64);b=torch.randn(3,2,256);b[:,:,:64]=a;b.requires_grad_()
        partial=torch.randn(3,2)
        torch.testing.assert_close(old(state,a,partial),new(state,b,partial),atol=1e-6,rtol=1e-5)
        new(state,b,partial).sum().backward();self.assertIsNone(b.grad)

    def test_mixed_historical_memory_sizes(self):
        models=[widen_actor(self.source()) for _ in range(2)]
        history=[[PersistentActor(),PersistentActor()]]
        physical=np.zeros((3,2,138),np.float32);physical[:,:,5]=1
        memory=[torch.randn(3,256) for _ in range(2)]
        roles=np.array([[-1,0],[0,-1],[-1,-1]])
        _,after,_,_=act_grouped(models,history,physical,memory,np.zeros((3,2,2),np.float32),roles,sample=False)
        obs=torch.zeros(1,140);obs[:,5]=1
        _,_,_,expected=history[0][1](obs,memory[1][0:1,:64])
        torch.testing.assert_close(after[1][0,:64],expected[0])
        self.assertEqual(after[1][0,64:].abs().sum(),0)

    def test_long_sequence_updates_are_finite(self):
        model=widen_actor(self.source());h,n=128,2
        obs=torch.randn(h,n,140);obs[:,:,5]=1
        starts=torch.zeros(h,n);starts[0]=1;starts[73]=1
        memories=torch.zeros(h,n,256);raw=torch.zeros(h,n,5);logp=torch.zeros(h,n)
        mem=torch.zeros(n,256)
        with torch.no_grad():
            for t in range(h):
                mem=mem*(1-starts[t,:,None]);memories[t]=mem
                _,_,raw[t],logp[t],_,mem=model.act(obs[t],mem)
        stats=actor_update(model,torch.optim.Adam(model.parameters(),lr=.0001),
            [obs,memories,starts,raw,logp],torch.randn(h,n),torch.ones(h,n,dtype=torch.bool),0,
            epochs=1,sequence_length=128,sequence_batch=2)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__=='__main__': unittest.main()
