def generate_images(autoencoder, K, n_images=1):
    """Generate n_images 'new' images from the decoder part of the given
    autoencoder.

    returns (n_images, channels, height, width) tensor of images
    """
    # Concatenate tuples to get (n_images, channels, height, width)
    output_shape = (n_images,) + my_dataset_size
    with torch.no_grad():
        z = torch.randn(n_images, K) # n_images along batch dimension
        x = autoencoder.decode(z).reshape(output_shape)
        return x

images = generate_images(conv_ae, K, n_images=25)

plt.figure(figsize=(5,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plot_torch_image(images[i])
plt.show()
