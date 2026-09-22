import numpy as np
from mab_utils import *

testbed = TestBed(3, [0, 5, 10], [1, 1, 1])
simple_bandit_agent = SimpleBandit(3, [0, 0, 0], 0.1)

learning_curve, action_log = bandit_simulation(simple_bandit_agent, testbed, time_steps = 100, epochs = 1)

print(learning_curve)
print(action_log)

