import numpy as np

def bandit_simulation(agent,
                      testbed,
                      time_steps = 1000,
                      epochs = 10):
    """
    :n_arms: number of arms
    :param agent:
    :param testbed:
    :param time_steps:
    :param epochs:
    :return:
    """

    testbed_log = {}
    action_log = np.empty((epochs, time_steps, 3))

    for e in range(epochs):
        agent.reset_agent()
        testbed.reset_testbed()

        testbed_means = testbed.means
        testbed_vars = testbed.variances
        testbed_log[e] = {'means': testbed_means,
                        'vars': testbed_vars,
                        'optimal_action': testbed.optimal_action}
        for t in range(time_steps):
            action = agent.choose_action()
            reward = testbed.pull_arm(action)
            agent.update_reward(action, reward)

            action_log[e][t] = (action, reward, action == testbed.optimal_action)

    return action_log,  testbed_log






