# Q-learning
def q_learning_backup(state, action, reward, next_state, values, params):
    '''
    Compute a new set of q-values using the q-learning update rule.
    Args:
        state (tuple): s_t, a tuple of xy coordinates.
        action (int): a_t, an integer from {0, 1, 2, 3}.
        reward (float): the reward of executing a_t at s_t.
        next_state (tuple): s_{t+1}, a tuple of xy coordinates.
        values (ndarray): an (h, w, 4) numpy array of q-values. values[y, x, a] 
                          stores the value of executing action a at state (x, y).
        params (dict): a dictionary of parameters.

    Returns:
        ndarray: the updated q-values.
    '''
    x, y = state
    nx, ny = next_state
    gamma = params['gamma']
    alpha = params['alpha']
    
    q = values[y, x, action]
    max_next_q = np.max(values[ny, nx])

    td_error = reward + (gamma * max_next_q - q)
    values[y, x, action] = q + alpha * td_error

    return values
