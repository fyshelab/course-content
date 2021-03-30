def run_poly_clasification(poly_degree):

  def make_poly_features(poly_degree, X):
    # Define the number of polynomial features except the bias term
    num_features = (poly_degree+1)*(poly_degree+2)//2-1
    poly_X = torch.zeros((X.shape[0], num_features))
    count = 0
    for i in range(poly_degree+1):
      for j in range(poly_degree+1):
         # no need to add zero degree since model has biases
        if j+i > 0:
          if j+i <= poly_degree:
            # Define the polynomial term
            poly_X[:, count] = X[:, 0]**i * X[:, 1]**j
            count += 1
    return poly_X, num_features

  poly_X_test, num_features = make_poly_features(poly_degree, X_test)
  poly_X_train, _ = make_poly_features(poly_degree, X_train)

  batch_size = 128
  poly_test_data = TensorDataset(poly_X_test, y_test)
  poly_test_loader = DataLoader(poly_test_data, batch_size=batch_size,
                          shuffle=False, num_workers=1)
  poly_train_data = TensorDataset(poly_X_train, y_train)
  poly_train_loader = DataLoader(poly_train_data, batch_size=batch_size,
                          shuffle=True, num_workers=1)
  
  # define a linear model using MLP class
  poly_net = Net('ReLU()', num_features, [], K).to(dev)

  # Train it!
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(poly_net.parameters(), lr=1e-3)
  _, _ = train_test_classification(poly_net, criterion, optimizer, 
                                  poly_train_loader, poly_test_loader,
                                  num_epochs=100)
  # Test it
  X_all = sample_grid().to(dev)
  poly_X_all, _ = make_poly_features(poly_degree, X_all)
  y_pred = poly_net(poly_X_all.to(dev))

  # Plot it
  with plt.xkcd():
    plot_decision_map(X_all, y_pred, X_test, y_test)
    plt.show()

  return num_features


max_poly_degree = 50
num_features = run_poly_clasification(max_poly_degree)
print('Number of features: %d'%num_features)
