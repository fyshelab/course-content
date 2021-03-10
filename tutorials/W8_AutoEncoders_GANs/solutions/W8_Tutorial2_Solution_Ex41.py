def augment_image_with_labels(img, labels):          
    onehot_z_target = torch.diag(torch.ones(num_classes))[labels].to(device)
    onehot_channel_target = F.interpolate(onehot_z_target.view(batch_size, -1, 1, 1), (H, W))
    augmented_img = torch.cat([img, onehot_channel_target], axis=1)
    return augmented_img
