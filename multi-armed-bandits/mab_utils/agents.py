import numpy as np

class SimpleBandit(object):
    def __init__(self, n_actions, rewards, epsilon):
        self.n_actions = n_actions
        self.action_steps = np.zeros(n_actions)
        self.epsilon = epsilon
        self.rewards = rewards
        self.rewards_init = rewards.copy()
        self.lifetime_reward = 0

    def choose_action(self):
        _random_number =np.random.random_sample()
        if _random_number < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.rewards)

        self.action_steps[action] += 1
        return action

    def update_reward(self, action, reward):
        self.rewards[action] = self.rewards[action] + (1/self.action_steps[action]) * (
            reward - self.rewards[action])
        self.lifetime_reward += reward
        return None

    def reset_agent(self):
        self.action_steps = np.zeros(self.n_actions)
        self.lifetime_reward = 0
        self.rewards = self.rewards_init.copy()

class ExponentialBandit(SimpleBandit):
    def __init__(self, n_actions, rewards, epsilon, alpha=0.1):
        SimpleBandit.__init__(self, n_actions, rewards, epsilon)
        self.alpha = alpha

    def update_reward(self, action, reward):
        self.rewards[action] = self.rewards[action] + self.alpha * (
            reward - self.rewards[action])
        self.lifetime_reward += reward
        return None
