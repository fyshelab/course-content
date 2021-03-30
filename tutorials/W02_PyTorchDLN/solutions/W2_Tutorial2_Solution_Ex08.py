def plot_distances(sample_size, dimensions):

  plt.figure(figsize=(15, 12))
  for cnt, D in enumerate(dimensions):

    norms = []

    if D != 1:
      mean = np.zeros(D)
      cov = 2*np.eye(D)  # diagonal covariance, i.e., isotropic gaussian
      x = np.random.multivariate_normal(mean, cov, size=sample_size).T
      x /= np.linalg.norm(x, axis=0).reshape(1,-1)
      y = np.random.multivariate_normal(mean, cov, size=sample_size).T
      y /= np.linalg.norm(y, axis=0).reshape(1,-1)
      norms = np.linalg.norm(x-y, axis=0, ord=2)
    
    elif D == 1:
      x = np.random.randn(sample_size)
      y = np.random.randn(sample_size)
      norms = np.abs(x - y)
    
    if cnt == 0:
      bins = np.histogram(norms, bins=100)[1]  # get the bin edges
    plt.subplot(3, 2, cnt + 1)
    plt.hist(norms, bins=bins, density=True, alpha=0.5)
    if D == 1:
      plt.xlabel('|x-y|')
    else:
      plt.xlabel('||x-y||')

    plt.ylabel('probability density')
    plt.title(f'{D}-D Gaussian')
      
  plt.tight_layout()


sample_size = 500
dimensions = [1, 2, 5, 10, 100, 1000]
plot_distances(sample_size)
