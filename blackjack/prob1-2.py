import numpy as np

gamma = 1
reward = np.array([20, -5, -5, -5, 100])
stop_thresh = 1e-3

# transit matrix for action +1
transit_mat_pos = np.array(
    [
    [.7, 0, .3, 0, 0],
    [0, .7, 0, .3, 0],
    [0, 0, .7, 0, .3],
    ])
#transit matrix for action -1
transit_mat_neg = np.array(
    [
    [.8, 0, .2, 0, 0],
    [0, .8, 0, .2, 0],
    [0, 0, .8, 0, .2],
    ])

# initialize each state's value to 0
Vopt = np.zeros(5)
iteration = 0

while True:
    iteration += 1
    Vopt_prev = Vopt.copy()
    Qpos = transit_mat_pos.dot(reward + gamma * Vopt_prev)
    Qneg = transit_mat_neg.dot(reward + gamma * Vopt_prev)
    Vopt[1:4] = np.maximum(Qpos, Qneg)
    print 'Iteration{}'.format(iteration), Vopt
    if np.all(np.abs(Vopt - Vopt_prev) < stop_thresh):
        print 'Stop at iteration {}'.format(iteration)
        break

opt_action = np.argmax(np.stack((Qneg, Qpos)), axis = 0) * 2 - 1
for state, action in zip([-1, 0, 1], opt_action):
    print 'Optimal action for state {:+d} is {:+d}'.format(state, action)