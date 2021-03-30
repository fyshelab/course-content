def convolution2d(image, kernel):
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y_op, x_op = image.shape
        y_op = y - m + 1
        x_op = x - m + 1
        convolved_image = np.zeros((y_op, x_op))
        for i in range(y_op):
            for j in range(x_op):
                convolved_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    
    return convolved_image


image = np.arange(9).reshape(3, 3)
print("Image:\n",image)
kernel = np.arange(4).reshape(2, 2)
print("Kernel:\n",kernel)
print("Convolved output:\n",convolution2d(image, kernel))
