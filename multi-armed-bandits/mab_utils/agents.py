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

class UcbBandit(SimpleBandit):
    def __init__(self, n_actions, rewards, c):
        super().__init__(n_actions, rewards, epsilon=0)  # epsilon unused, so set to 0
        self.c = c
        self.t = 0  # total step counter

    def choose_action(self):
        self.t += 1

        zero_actions = np.where(self.action_steps == 0)[0]

        if len(zero_actions) > 0:
            action = np.random.choice(zero_actions)
            self.action_steps[action] += 1

        else:
            confidence_adj_rewards = self.rewards + self.c * (np.sqrt(np.log(self.t)/self.action_steps))
            action = np.argmax(confidence_adj_rewards)
            self.action_steps[action] += 1

        return action

    def reset_agent(self):
        self.action_steps = np.zeros(self.n_actions)
        self.lifetime_reward = 0
        self.rewards = self.rewards_init.copy()
        self.t = 0

class GradientBandit(SimpleBandit):
    def __init__(self, n_actions, rewards, preferences, alpha=0.1):
        super().__init__(n_actions, rewards, epsilon=0)  # epsilon unused, so set to 0
        self.preferences = preferences
        self.preferences_init = preferences.copy()
        self.alpha = alpha

    def softmax(self):
        return np.exp(self.preferences)/np.sum(np.exp(self.preferences))

    def choose_action(self):
        action = np.argmax(self.preferences)
        self.action_steps[action] += 1

        return action

    def update_preferences(self, action, reward):
        probs = self.softmax()

        for a in range(self.n_actions):
            if a == action:
                self.preferences[a] = self.preferences[a] + self.alpha * (
                        reward - self.rewards[a]) * (1 - probs[a]
                )

            else:
                self.preferences[a] = self.preferences[a] - self.alpha * (
                        reward - self.rewards[a]) * probs[a]
        return None

    def update_reward(self, action, reward):
        self.rewards[action] = self.rewards[action] + (1 / self.action_steps[action]) * (
                reward - self.rewards[action])
        self.lifetime_reward += reward

        # Update preferences. Need to update sim tools to remove update_reward call to abstract this
        self.update_preferences(action, reward)

        return None

    def reset_agent(self):
        self.action_steps = np.zeros(self.n_actions)
        self.lifetime_reward = 0
        self.rewards = self.rewards_init.copy()
        self.preferences = self.preferences_init.copy()
