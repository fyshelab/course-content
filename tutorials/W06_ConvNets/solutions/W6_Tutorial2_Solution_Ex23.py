def get_augmentation_transforms():
   
    augmentation_trasnforms = [transforms.RandomRotation(10), transforms.RandomHorizontalFlip()]

    return augmentation_trasnforms 
