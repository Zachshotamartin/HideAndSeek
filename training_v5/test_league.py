import copy
import unittest
import numpy as np
import torch
from persistent_actor import PersistentActor, augment
from league_ppo import assign_roles, active_masks, act_grouped, actor_update
from physics import PhysicsEnv

torch.set_num_threads(1)


class LeagueTests(unittest.TestCase):
    def test_assignment_and_sample_counters_distinguish_current_frozen_and_blind(self):
        first=np.random.default_rng(733);second=np.random.default_rng(733)
        a=assign_roles(first,4000,'A',3);b=assign_roles(second,4000,'B',3)
        self.assertTrue((a==-1).all());self.assertFalse(((b>=0).all(axis=1)).any())
        self.assertEqual(first.bit_generator.state,second.bit_generator.state)
        self.assertTrue(.77 < (b==-1).all(1).mean() < .83)
        physical=np.zeros((4000,2,208),np.float32);physical[:2000,1,5]=1
        current,active=active_masks(physical,b)
        self.assertEqual(int(current.sum()+(~current).sum()),8000)
        np.testing.assert_array_equal(active[:,0],current[:,0])
        self.assertFalse(active[2000:,1].any())
        self.assertTrue((active <= current).all())

    def test_grouped_actual_actor_identity_values_memory_and_blind_reset(self):
        torch.manual_seed(22)
        models=[PersistentActor(),PersistentActor()]
        history=[[PersistentActor().eval().requires_grad_(False),PersistentActor().eval().requires_grad_(False)]]
        physical=np.stack([PhysicsEnv(seed=831).observe(),PhysicsEnv(seed=991).observe()])
        roles=np.array([[-1,0],[0,-1]])
        memories=[torch.randn(2,64),torch.randn(2,64)]
        buttons=np.zeros((2,2,2),np.float32)
        rng=torch.get_rng_state().clone()
        _,next_memory,_,_,records=act_grouped(models,history,physical,memories,buttons,roles,sample=False)
        torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
        observed=augment(physical,buttons)
        with torch.no_grad():
            for role in range(2):
                for row in range(2):
                    actor=models[role] if roles[row,role]==-1 else history[0][role]
                    _,_,value,memory=actor(torch.from_numpy(observed[row:row+1,role]),memories[role][row:row+1])
                    torch.testing.assert_close(records[role][3][row],value[0])
                    torch.testing.assert_close(next_memory[role][row],memory[0])
        actions,new_memory,new_buttons,_,_=act_grouped(models,history,physical,memories,buttons,roles)
        np.testing.assert_array_equal(actions[:,1],0);np.testing.assert_array_equal(new_buttons[:,1],0)
        self.assertTrue(any(not torch.equal(old,new) for old,new in zip(memories,new_memory)))
        self.assertTrue(all(p.grad is None for pair in history for actor in pair for p in actor.parameters()))

    def test_all_frozen_batch_skips_adam_momentum_and_never_reads_extreme_logps(self):
        model=PersistentActor();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        model.encoder.weight.sum().backward();optimizer.step();before=copy.deepcopy(model.state_dict());oldopt=copy.deepcopy(optimizer.state_dict())
        batch=[torch.randn(4,2,210),torch.randn(4,2,64),torch.zeros(4,2),torch.full((4,2,5),1e20),torch.full((4,2),float('nan'))]
        stats=actor_update(model,optimizer,batch,torch.full((4,2),float('nan')),torch.zeros(4,2,dtype=torch.bool),0,sequence_length=4)
        self.assertEqual(stats['optimizerSteps'],0);self.assertEqual(stats['activeSamples'],0)
        for name,value in model.state_dict().items():torch.testing.assert_close(value,before[name],atol=0,rtol=0)
        for key,state in optimizer.state_dict()['state'].items():
            for name,value in state.items():torch.testing.assert_close(value,oldopt['state'][key][name],atol=0,rtol=0)

    def test_historical_episode_prefix_has_no_gradient_path_after_current_reset(self):
        torch.manual_seed(92)
        model=PersistentActor();other=copy.deepcopy(model)
        observed=torch.randn(4,1,210);observed[:,:,5]=1
        raw=torch.randn(4,1,5);raw[:,:,3:]=0
        starts=torch.zeros(4,1);starts[2]=1
        memories=torch.randn(4,1,64);memories[2]=0
        logps=torch.zeros(4,1);logps[:2]=float('nan')
        advantages=torch.tensor([[float('nan')],[float('nan')],[.5],[-.5]])
        full=[observed,memories,starts,raw,logps]
        short=[item[2:].clone() for item in full]
        torch.manual_seed(991)
        left=actor_update(model,torch.optim.Adam(model.parameters(),lr=.0001),full,advantages,
            torch.tensor([[False],[False],[True],[True]]),1,epochs=1,sequence_length=4,entropy_weight=0)
        torch.manual_seed(991)
        right=actor_update(other,torch.optim.Adam(other.parameters(),lr=.0001),short,advantages[2:],
            torch.ones(2,1,dtype=torch.bool),1,epochs=1,sequence_length=2,entropy_weight=0)
        self.assertAlmostEqual(left['loss'],right['loss'],places=6)
        self.assertAlmostEqual(left['approximateKL'],right['approximateKL'],places=6)
        for name,value in model.state_dict().items():torch.testing.assert_close(value,other.state_dict()[name],atol=1e-7,rtol=1e-6)

    def test_burn_in_runs_through_prefix_and_stored_memories(self):
        class Counting(PersistentActor):
            calls=0
            def forward(self,o,m): Counting.calls+=1; return super().forward(o,m)
        def batch(h,n):
            observed=torch.randn(h,n,210); observed[:,:,5]=1
            raw=torch.randn(h,n,6); raw[:,:,4:]=0
            return [observed,torch.randn(h,n,64),torch.zeros(h,n),raw,torch.zeros(h,n)]
        def update(h,seq,burn,prefix):
            torch.manual_seed(5); model=Counting(); Counting.calls=0
            stats=actor_update(model,torch.optim.Adam(model.parameters(),lr=1e-4),batch(h,2),torch.randn(h,2),torch.ones(h,2,dtype=torch.bool),0,
                epochs=1,sequence_length=seq,sequence_batch=2,burn_in=burn,prefix=prefix)
            return Counting.calls,stats['burnInSteps']
        prefix=(torch.randn(4,2,210),torch.randn(4,2,64),torch.zeros(4,2))
        # Without a prefix the first sequence cannot be warmed; with one, every sequence is.
        self.assertEqual(update(16,16,4,None),(16,0))
        self.assertEqual(update(16,16,4,prefix),(20,8))
        self.assertEqual(update(16,8,4,None),(20,8))
        self.assertEqual(update(16,8,4,prefix),(24,16))
        with self.assertRaises(ValueError):update(16,16,4,(torch.randn(3,2,210),torch.randn(3,2,64),torch.zeros(3,2)))
    def test_burn_in_prefix_yields_recomputed_start_memory(self):
        torch.manual_seed(9); model=PersistentActor()
        prefix=(torch.randn(3,1,210),torch.randn(3,1,64),torch.tensor([[1.],[0.],[0.]]))
        seen=[]
        original=model.forward
        def spy(o,m): seen.append(m.clone()); return original(o,m)
        model.forward=spy
        observed=torch.randn(2,1,210); observed[:,:,5]=1; raw=torch.randn(2,1,6); raw[:,:,4:]=0
        actor_update(model,torch.optim.Adam(model.parameters(),lr=0.),[observed,torch.randn(2,1,64),torch.zeros(2,1),raw,torch.zeros(2,1)],torch.ones(2,1),
            torch.ones(2,1,dtype=torch.bool),0,epochs=1,sequence_length=2,sequence_batch=1,burn_in=3,prefix=prefix)
        with torch.no_grad():
            memory=torch.zeros(1,64)  # the prefix starts an episode, so its stored memory is discarded
            for step in range(3):memory=original(prefix[0][step],memory*(1-prefix[2][step][:,None]))[-1]
        torch.testing.assert_close(seen[3],memory)

if __name__=='__main__':unittest.main()
