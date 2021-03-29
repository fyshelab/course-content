# Random Policy evaluation
def random_policy_evaluation(grid, gamma=1.0):
    values = np.zeros_like(grid.rew_grid)
    iter = 0
    while True:
        eps = 0
        for y in range(grid.h):
            for x in range(grid.w):
                v = values[y, x]
                new_v = 0
                for action in range(grid.n_actions):
                    (new_x, new_y), reward = grid.get_transition((x, y), action)
                    new_v += 0.25 * (reward + gamma * values[new_y, new_x])
                values[y, x] = new_v
                eps = max(eps, abs(new_v - v))
        iter += 1
        if eps < 0.0001:
            print("Converged after {} iterations".format(iter))
            break
    return values
