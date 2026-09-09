import unittest
import numpy as np
from sim import BatchArena,generate_arena
class SimulationTests(unittest.TestCase):
    def test_reset_does_not_age_other_arenas(self):
        env=BatchArena(3,123,1);env.age[1]=[17,19];env.reset(0);self.assertEqual(env.age[1].tolist(),[17,19])
    def test_hidden_opponent_invariance(self):
        arena={'seed':1,'width':12,'height':12,'blocks':[{'x':5,'y':3,'width':1,'height':3},{'x':5,'y':6,'width':1,'height':2}],'spawns':[[3.5,5.5],[7.5,5.5]]}
        env=BatchArena(1);env.reset(0,arena);env.t[:]=30;env.update_sight(age=False);before=env.observe()[0,1].copy();env.pos[0,0]=[3.5,6.5];env.update_sight(age=False);np.testing.assert_array_equal(before,env.observe()[0,1])
    def test_prep_blind(self):
        env=BatchArena(1);arena,_=generate_arena(3,12,12,0);env.reset(0,arena);start=env.pos[0,1].copy()
        for _ in range(24):
            self.assertFalse(env.known[0,1]);np.testing.assert_array_equal(env.observe()[0,1,8:14],0);env.step(np.array([[0,2]]),False);np.testing.assert_array_equal(env.pos[0,1],start)
if __name__=='__main__':unittest.main()
