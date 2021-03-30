def get_gen_modified_BCE_loss(disc_fake_pred):
    '''
    Function for returning a modified BCE loss for generator

    Parameters:
        probs of real predicted by the disc
        using a fake batch, (num_samples, 1)

    Returns:
        modified BCE loss of the fake prediction batch, a scalar
    '''
    
    # modified BCE loss of the fake batch
    gen_fake_loss = -torch.mean(torch.clamp(torch.log(disc_fake_pred), -100, 0))

    return gen_fake_loss
