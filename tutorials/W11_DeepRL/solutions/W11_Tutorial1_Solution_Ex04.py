# Q-learning
def q_learning_backup(state, action, reward, next_state, values, params):
    x, y = state
    nx, ny = next_state
    gamma = params['gamma']
    alpha = params['alpha']
    
    q = values[y, x, action]
    max_next_q = np.max(values[ny, nx])

    td_error = reward + (gamma * max_next_q - q)
    values[y, x, action] = q + alpha * td_error

    return values
