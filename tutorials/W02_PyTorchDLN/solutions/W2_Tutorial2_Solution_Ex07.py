def criterion(x):
  margin = 0.5
  loss = torch.mean(-F.cosine_similarity(x[:,0], x[:,1]) +
                    F.cosine_similarity(x[:,0], x[:,2]) +
                    margin)
  
  return loss


x = torch.tensor([[[1., 2.], [3., 2.], [9., 2.]],
                  [[-1., 3.], [4., 2.], [8., 2.]],
                  [[1., 1.], [-2., 1.], [7., 4.]]])
print(x)

print(criterion(x))
