def get_disc_BCE_loss(disc_real_pred, disc_fake_pred):
    '''
    Function for returning a BCE loss of discriminator

    Parameters:
        disc_real_pred: probs of real predicted by the disc
        using a real batch, (num_samples, 1)
        probs of real predicted by the disc
        using a fake batch, (num_samples, 1)

    Returns:
        BCE loss of the two prediction batches, a scalar
    '''

    # BCE loss of the real batch
    disc_real_loss = -torch.mean(torch.clamp(torch.log(disc_real_pred), -100, 0))

    # BCE loss of the fake batch
    disc_fake_loss = -torch.mean(torch.clamp(torch.log(1 - disc_fake_pred), -100, 0))

    # Average loss
    disc_loss = (disc_real_loss + disc_fake_loss) / 2
    return disc_loss
