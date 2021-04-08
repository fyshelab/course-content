def update_q_value_fixed(actions):
    # p (policy) decides the probability with each action is selected. As the
    # probability of one action becomes larger, that action is selected more often.
    # The probability of an action is stored in actions[i].value
    num_actions = len(actions)
    probs = [e.value for e in actions]
    i = np.random.choice(list(range(num_actions)), p=probs)

    # Noise is added to simulate values we get in training which vary with every episode
    q_value_est = actions[i].q_val + np.random.randn() * actions[i].q_val

    # Complete the update rule. alpha is given by lr. The gradient and probabiliy 
    # of an action can be accessed via actions[i].grad and actions[i].value.
    actions[i].logit += lr * q_value_est * actions[i].grad / actions[i].value
    
    return i, lr * q_value_est / actions[i].value / 10
