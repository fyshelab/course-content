def update_q_value_fixed(actions):

    i = np.random.choice(list(range(len(actions))), p=[e.value for e in actions])

    q_value_est = actions[i].q_val + np.random.randn() * actions[i].q_val

    actions[i].logit += lr * q_value_est * actions[i].grad / actions[i].value
    
    return i, lr * q_value_est / actions[i].value / 10
