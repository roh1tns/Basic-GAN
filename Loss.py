from Generator import *

# Calculating the loss

# Generator Loss


def calc_gen_loss(loss_func, gen, disc, number, z_dim):
    noise = gen_noise(number, z_dim)
    fake = gen(noise)
    pred = disc(fake)
    target = torch.ones_like(pred)
    gen_loss = loss_func(pred, target)
    return gen_loss


# Discriminator loss

def calc_disc_loss(loss_func, gen, disc, number, real, z_dim):
    noise = gen_noise(number, z_dim)
    fake = gen(noise)
    disc_fake = disc(fake.detach())
    disc_fake_target = torch.zeros_like(disc_fake)
    disc_fake_loss = loss_func(disc_fake, disc_fake_target)

    disc_real = disc(real)
    disc_real_target = torch.ones_like(disc_real)
    disc_real_loss = loss_func(disc_real, disc_real_target)

    disc_loss = (disc_fake_loss + disc_real_loss)

    return disc_loss
