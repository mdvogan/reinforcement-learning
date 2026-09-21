import numpy as np
from mab_utils import *

test_bed = TestBed(3, [0, 5, 10], [1, 1, 1])

for i in range(test_bed.n_arms):
    print(f"Bed {i} sample: {test_bed.pull_arm(i, 1)}")