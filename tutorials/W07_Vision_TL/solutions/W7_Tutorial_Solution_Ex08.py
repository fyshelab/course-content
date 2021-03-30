def train_one_epoch_with_replay(model: nn.Module,
                                pokemon_train_loader: torch.utils.data.DataLoader,
                                imagenette_train_loader: torch.utils.data.DataLoader,
                                pokemon_classifier_head: nn.Module,
                                imagenette_classifier_head: nn.Module,
                                pokemon_optimizer: torch.optim.Optimizer,
                                imagenette_optimizer: torch.optim.Optimizer
                                ):
    """
    Write the code required to train for one epoch on both the pokemon and imagenette datasets.
    This is called "replay" because you continue training on old data. In practice replay would be
    done less often that every epoch, but alternating epochs of pokemon and imagenette training
    works for our purposes.

    Tips: The `train_one_epoch_pokemon` and `train_one_epoch_imagenette` functions will be useful 
          here.
          Don't forget to use `model.fc = pokemon_classifier_head` or `imagenette_classifier_head`
          before training on their respective datasets.

    Parameters:
        model: The model to train
        pokemon_train_loader: The DataLoader with pokemon data to be used in training
        imagenette_train_loader: The DataLoader with image data to be used in training
        pokemon_classifier_head: The fully connected layer used for predicting which pokemon is 
                                 present in an image
        imagenette_classifier_head: The fully connected layer used to predict which imagenette
                                    class is present in an image
        pokemon_optimizer: The optimizer used to update the weights of the resnet and pokemon 
                           classifier head
        imagenette_optimizer: The optimizer used to update the weights of the resnet and imagenette
                              classifier head
    
    Returns:
        None
    """

    model.fc = pokemon_classifier_head
    model.train()

    train_one_epoch_pokemon(model, pokemon_train_loader, pokemon_optimizer, loss_fn)

    # Set model to use the imagenette classifier head
    model.fc = imagenette_classifier_head
    model.train()

    train_one_epoch_imagenette(model, imagenette_train_loader, imagenette_optimizer, loss_fn)


### Uncomment below to test your function
top_1_accs = []
top_5_accs = []
pokemon_accs = []

for epoch in range(5):
    # Set model to use the pokemon classification output layer
    
    train_one_epoch_with_replay(resnet, pokemon_train_loader, imagenette_train_loader, 
                                pokemon_classifier_head, imagenette_classifier_head, 
                                pokemon_optimizer, imagenette_optimizer)

    resnet.eval()
    top_1, top_5 = eval_imagenette(resnet, imagenette_val_loader, len(imagenette_val))
    top_1_accs.append(top_1.item())
    top_5_accs.append(top_5.item())

    # Evaluate model on pokemon 
    resnet.eval()
    resnet.fc = pokemon_classifier_head
    pokemon_acc = eval_pokemon(resnet, pokemon_test_loader, len(pokemon_test_set))
    pokemon_accs.append(pokemon_acc)
    
    plt.plot(top_1_accs, label='Top-1 Accuracy')
    plt.plot(top_5_accs, label='Top-5 Accuracy')
    plt.plot(pokemon_accs, label='Pokemon Prediction Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Preventing Catastrophic Forgetting')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
