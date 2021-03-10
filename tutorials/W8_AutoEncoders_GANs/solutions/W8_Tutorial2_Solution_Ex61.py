def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        real_X: the real images from pile X
        fake_X: the generated images of class X
        disc_X: the discriminator for class X; takes images and returns real/fake class X
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the target labels and returns a adversarial 
            loss (which you aim to minimize)
    '''
    disc_fake_X_pred = disc_X(fake_X.detach()) # Detach generator
    disc_fake_X_loss = adv_criterion(disc_fake_X_pred, torch.zeros_like(disc_fake_X_pred))
    disc_real_X_pred = disc_X(real_X)
    disc_real_X_loss = adv_criterion(disc_real_X_pred, torch.ones_like(disc_real_X_pred))
    disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2
    return disc_loss
