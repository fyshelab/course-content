# Value Iteration
def value_iteration(grid, gamma=0.9):
    V = np.zeros_like(grid.rew_grid)
    while True:
        eps = 0
        for y in range(grid.h):
            for x in range(grid.w):
                v = values[y, x]
                action_values = np.zeros(grid.n_actions)
                for action in range(grid.n_actions):
                    (nx, ny), reward = grid.get_transition((x, y), action)
                    action_values[action] = reward + gamma * values[ny, nx] 
                new_v = np.max(action_values)
                values[y, x] = new_v
                eps = max(eps, abs(new_v - v))
        if eps < 0.0001:
            break

    # Create greedy policy from values
    policy = np.zeros_like(grid.rew_grid).astype(int)
    for y in range(grid.h):
        for x in range(grid.w):
            action_values = np.zeros(grid.n_actions)
            for action in range(grid.n_actions):
                (nx, ny), reward = grid.get_transition((x, y), action)
                action_values[action] = reward + gamma * values[ny, nx] 
            policy[y, x] = np.argmax(action_values)

    return values, policy
