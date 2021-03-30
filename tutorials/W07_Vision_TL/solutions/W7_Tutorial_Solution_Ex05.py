def calculate_pairwise_distances(embedding_tensor: torch.tensor):
    """
    This function calculates the pairwise distance between each image embedding in a tensor

    Parameters:
        embedding_tensor: A num_images x embedding_dimension tensor

    Returns:
        distances: A num_images x num_images tensor containing the pairwise distances between each 
                   image embedding

    Hint: the function torch.cdist makes this exercise a one-liner, though there are several ways to
          calculate pairwise distances
    """
    distances = torch.cdist(embedding_tensor, embedding_tensor)

    return distances

distances = calculate_pairwise_distances(embedding_tensor)

plt.figure(figsize=(8,8))
plt.imshow(distances.detach().numpy())
plt.annotate('Bruce', (3,-.5), fontsize=24, va='bottom')
plt.annotate('Neil', (20,-.5), fontsize=24, va='bottom')
plt.annotate('Pam', (35,-.5), fontsize=24, va='bottom')
plt.annotate('Bruce', (-.5,10), fontsize=24, rotation=90, ha='right')
plt.annotate('Neil', (-.5,24), fontsize=24, rotation=90, ha='right')
plt.annotate('Pam', (-.5,39), fontsize=24, rotation=90, ha='right')
plt.colorbar()
plt.axis('off')
