def create_spiral_dataset(K, sigma, N):

    # Initialize t, X, y
    t = torch.linspace(0, 1, N)
    X = torch.zeros(K*N, 2)
    y = torch.zeros(K*N)

    # Create data
    for k in range(K):
      X[k*N:(k+1)*N, 0] = t*(torch.sin(2*np.pi/K*(2*t+k)) + sigma*torch.randn(N))   
      X[k*N:(k+1)*N, 1] = t*(torch.cos(2*np.pi/K*(2*t+k)) + sigma*torch.randn(N))   
      y[k*N:(k+1)*N] = k   

    return X, y

# Set parameters
K = 4
sigma = 0.16
N = 1000
 
X, y = create_spiral_dataset(K, sigma, N)

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.show()
