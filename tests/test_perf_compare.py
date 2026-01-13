import os
import time
import random
import unittest
import numpy as np

from fuzzy_logic import decide_use_nn

@unittest.skipUnless(os.path.exists('./saved_models/rl_agent_snr_threshold.pkl'), "RL model not available")
class TestPerformanceCompare(unittest.TestCase):
    def test_decision_speed(self):
        from rl_agent import RLAgent
        rl_agent = RLAgent(
            state_dim=3,
            action_dim=10,
            lr=0.001,
            gamma=0.99,
            epsilon_start=0.01,
            epsilon_end=0.01,
            epsilon_decay=1.0,
            device='cpu'
        )
        rl_agent.load_model('./saved_models/rl_agent_snr_threshold.pkl')

        # Generate random normalized states for Fuzzy Logic (0-1)
        states_fuzzy = [
            (random.random(), random.random(), random.random())
            for _ in range(1000)
        ]
        
        # Generate random raw states for RL Agent (Traditional ranges)
        # SNR: 20-40, Dist: 10-100, Speed: 0-25
        states_rl = [
            (
                random.uniform(20, 40),
                random.uniform(10, 100),
                random.uniform(0, 25)
            )
            for _ in range(1000)
        ]

        t0 = time.perf_counter()
        for s in states_fuzzy:
            _ = decide_use_nn(*s)
        fuzzy_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        for s in states_rl:
            # RL agent expects numpy array or list
            a = rl_agent.select_action(np.array(s), training=False)
            _ = rl_agent.get_snr_threshold(a)
        rl_time = time.perf_counter() - t1

        print(f"Fuzzy decision time: {fuzzy_time:.6f}s for 1000 states")
        print(f"RL decision time:    {rl_time:.6f}s for 1000 states")
        
        # Fuzzy logic should be significantly faster (simple math vs neural net)
        self.assertLess(fuzzy_time, rl_time)

if __name__ == '__main__':
    unittest.main()
