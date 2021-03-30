def augment_image_with_labels(img, labels):
    '''
    Function to return conditional input on the labels.

    Parameters:
        img: img batch tensor for discriminator, (batch_size, C, H, W)
        labels: the batch labels with 0 to C-1 values, (batch_size)

    Returns:
      extended input with one-hot labels, (batch_size, C + num_classes, H, W)
    '''                      
    onehot_z_target = torch.diag(torch.ones(num_classes))[labels].to(device)
    onehot_channel_target = F.interpolate(onehot_z_target.view(batch_size, -1, 1, 1), (H, W))
    augmented_img = torch.cat([img, onehot_channel_target], axis=1)
    return augmented_img
