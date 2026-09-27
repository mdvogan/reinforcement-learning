import numpy as np

def bandit_simulation(n_arms,
                      agent,
                      testbed_obj,
                      testbed_param_fun,
                      time_steps = 1000,
                      epochs = 10):
    """
    :n_arms: number of arms
    :param agent:
    :param testbed_obj:
    :param testbed_param_fun
    :param time_steps:
    :param epochs:
    :return:
    """

    testbed_log = {}
    action_log = np.empty((epochs, time_steps, 3))

    for e in range(epochs):
        print(f"Running Epoch: {e}")
        agent.reset_agent()


        testbed_means, testbed_vars = testbed_param_fun(n_arms)
        testbed = testbed_obj(len(testbed_means), testbed_means, testbed_vars)
        testbed_log[e] = {'means': testbed_means,
                        'vars': testbed_vars,
                        'optimal_action': testbed.optimal_action}
        for t in range(time_steps):
            action = agent.choose_action()
            reward = testbed.pull_arm(action)
            agent.update_reward(action, reward)

            action_log[e][t] = (action, reward, action == testbed.optimal_action)

    return action_log,  testbed_log






