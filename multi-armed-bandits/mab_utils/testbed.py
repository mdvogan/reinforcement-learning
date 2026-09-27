import numpy as np

class TestBed(object):
    def __init__(self, n_arms):
        self.n_arms = n_arms
        means, variances = self.generate_arm_moments(n_arms)
        self.means = means
        self.variances = variances
        self.optimal_action = np.argmax(self.means)
        self.arms = []
        for i in range(n_arms):
            self.arms.append((self.means[i], self.variances[i]))

    def generate_arm_moments(self, n_arms):
        means = np.random.normal(0, 1, n_arms)
        vars = np.ones(n_arms)
        return means, vars

    def pull_arm(self, arm, n = 1):
        return np.random.normal(self.arms[arm][0], self.arms[arm][1], n)[0]

    def reset_testbed(self):
        means, variances = self.generate_arm_moments(self.n_arms)
        self.means = means
        self.variances = variances
        self.optimal_action = np.argmax(self.means)
        self.arms = []
        for i in range(self.n_arms):
            self.arms.append((self.means[i], self.variances[i]))