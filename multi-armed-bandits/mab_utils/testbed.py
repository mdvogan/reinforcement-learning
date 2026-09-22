import numpy as np

class TestBed(object):
    def __init__(self, n_arms, means, variances):
        self.n_arms = n_arms
        self.means = means
        self.variances = variances
        self.optimal_action = np.argmax(self.means)
        self.arms = []
        for i in range(n_arms):
            self.arms.append((self.means[i], self.variances[i]))

    def pull_arm(self, arm, n = 1):
        return np.random.normal(self.arms[arm][0], self.arms[arm][1], n)[0]