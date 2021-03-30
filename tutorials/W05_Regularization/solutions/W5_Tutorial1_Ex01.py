def calculate_frobenius_norm(model):

    ####################################################################
    # Fill in all missing code below (...),
    # then remove or comment the line below to test your function
    # raise NotImplementedError("Define the grad visualization function")
    ####################################################################

    norm = 0.0

    # Sum all the parameters
    for param in model.parameters():
        norm += torch.sum(param**2)

    # Take a square root of the sum of squares of all the parameters
    norm = norm**0.5
    return norm

net = nn.Linear(10,1)
print(f'Frobenius Norm of Single Linear Layer: {calculate_frobenius_norm(net)}')
