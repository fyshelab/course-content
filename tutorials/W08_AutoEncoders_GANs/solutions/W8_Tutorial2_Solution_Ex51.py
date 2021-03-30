def calculate_updated_noise(noise, step_size):
    '''
    Function to return noise vectors updated with stochastic gradient ascent.
    Parameters:
        noise: the current noise vectors. You have already called the backwards function on the target class
          so you can access the gradient of the output class with respect to the noise by using noise.grad
        step_size: the scalar amount by which you should weight the noise gradient
    '''
    new_noise = noise + ( noise.grad * step_size)
    return new_noise
