# Policy Iteration 
def policy_evaluation(grid, values, policy, gamma):
    while True:
        eps = 0
        for y in range(grid.h):
            for x in range(grid.w):
                v = values[y, x]
                (new_x, new_y), reward = grid.get_transition((x, y), policy[y, x])
                new_v = reward + gamma * values[new_y, new_x]
                values[y, x] = new_v
                eps = max(eps, abs(new_v - v))
        if eps < 0.0001: 
            break

def policy_improvement(grid, values, policy, gamma):
    converged = True
    for y in range(grid.h):
        for x in range(grid.w):
            old_action = policy[y, x]
            action_values = np.zeros(grid.n_actions, dtype=np.float)
            for action in range(grid.n_actions):
                (new_x, new_y), reward = grid.get_transition((x, y), action)
                action_values[action] = reward + gamma * values[new_y, new_x]
            policy[y, x] = np.argmax(action_values)
            if old_action != policy[y, x]:
                converged = False
    return converged

def policy_iteration(grid, gamma=0.9):
    policy = np.random.choice(grid.n_actions, (grid.h, grid.w)).astype(int)
    values = np.zeros_like(grid.rew_grid)
    converged = False
    while not converged:
        print("running policy evaluation")
        policy_evaluation(grid, values, policy, gamma)
        print("running policy improvement")
        converged = policy_improvement(grid, values, policy, gamma)
    return values, policy
