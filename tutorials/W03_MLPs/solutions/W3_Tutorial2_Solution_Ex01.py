def run_grad_viz():
  # define a network which is randomly initialized, any ReLU model would do the trick!
  rand_net = Net('ReLU()', X_train.shape[1], [128], K).to(dev)

  # sample the grid to generate an input batch, hover over the tensor to see it's shape
  X_all = sample_grid(x_max=1).to(dev)

  # we need to allow PyTorch to keep track of the input gradients
  X_all.requires_grad = True

  # generate the output
  outputs = rand_net(X_all)

  # create a scaler value for backward()
  loss = torch.mean(outputs)

  # do the backgrop
  loss.backward()

  # defines a color where gradient is constant
  grad_colors = color_grad(X_all.grad)

  return grad_colors


grad_colors = run_grad_viz()
plt.imshow(grad_colors, cmap='rainbow')
plt.show()
