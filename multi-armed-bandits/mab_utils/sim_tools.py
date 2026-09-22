import numpy as np

def bandit_simulation(agent, testbed, time_steps = 1000, epochs = 10):
    """
    :param agent:
    :param testbed:
    :param time_steps:
    :param epochs:
    :return:
    """

    learning_curve = np.empty((epochs, time_steps, testbed.n_arms))
    action_log = np.empty((epochs, time_steps, 3))

    for e in range(epochs):
        for t in range(time_steps):
            action = agent.choose_action()
            reward = testbed.pull_arm(action)
            agent.update_reward(action, reward)

            learning_curve[e][t] = agent.rewards
            action_log[e][t] = (action, reward, action == testbed.optimal_action)

    return learning_curve, action_log






